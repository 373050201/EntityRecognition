"""公平流水线两个独立RT-DETRv2模型的训练损失。"""
import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from utils import CTC_BLANK_INDEX, encode_identifiers



PROJECT_ROOT = Path(__file__).resolve().parents[2]
RTDETR_ROOT = PROJECT_ROOT / "third_party" / "rtdetrv2"
if str(RTDETR_ROOT) not in sys.path:
    sys.path.insert(0, str(RTDETR_ROOT))
from src.zoo.rtdetr.matcher import HungarianMatcher  # noqa: E402
from src.zoo.rtdetr.rtdetrv2_criterion import RTDETRCriterionv2  # noqa: E402


def _build_detector_criterion(num_classes, bbox_coef, giou_coef):
    """构建官方RT-DETRv2匹配器与检测损失。"""
    matcher = HungarianMatcher(
        weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
        use_focal_loss=True,
        alpha=0.25,
        gamma=2.0,
    )
    criterion = RTDETRCriterionv2(
        matcher=matcher,
        weight_dict={
            "loss_vfl": 1.0,
            "loss_bbox": bbox_coef,
            "loss_giou": giou_coef,
        },
        losses=["vfl", "boxes"],
        alpha=0.75,
        gamma=2.0,
        num_classes=num_classes,
    )
    return matcher, criterion


def _get_src_permutation_idx(indices, device):
    """把每张图的匹配query索引拼接为批索引和query索引。"""
    batch_indices = torch.cat(
        [
            torch.full_like(query_indices, batch_index)
            for batch_index, (query_indices, _) in enumerate(indices)
        ]
    ).to(device)
    query_indices = torch.cat(
        [query_indices for query_indices, _ in indices]
    ).to(device)
    return batch_indices, query_indices


def _sum_detection_auxiliary_losses(detector_losses, outputs):
    """汇总RT-DETRv2中间层、编码器和去噪辅助损失。"""
    primary_names = {"loss_vfl", "loss_bbox", "loss_giou"}
    return sum(
        (
            loss
            for name, loss in detector_losses.items()
            if name not in primary_names
        ),
        outputs["pred_logits"].sum() * 0,
    )


class SymbolCriterion(nn.Module):
    """组合图元检测损失和匹配图元的锁状态损失。"""

    def __init__(
        self,
        num_classes=4,
        bbox_coef=5.0,
        giou_coef=2.0,
        locked_coef=1.0,
    ):
        super().__init__()
        self.matcher_impl, self.detector_criterion = _build_detector_criterion(
            num_classes,
            bbox_coef,
            giou_coef,
        )
        self.locked_coef = locked_coef

    @staticmethod
    def _prepare_targets(targets, device):
        return [
            {
                "labels": target["symbol_labels"].to(device),
                "boxes": target["symbol_boxes"].to(device),
            }
            for target in targets
        ]

    @torch.no_grad()
    def matcher(self, outputs, targets):
        detector_targets = self._prepare_targets(
            targets,
            outputs["pred_logits"].device,
        )
        return self.matcher_impl(outputs, detector_targets)["indices"]

    def forward(self, outputs, targets):
        device = outputs["pred_logits"].device
        detector_targets = self._prepare_targets(targets, device)
        detector_losses = self.detector_criterion(outputs, detector_targets)
        indices = self.matcher(outputs, targets)
        batch_indices, query_indices = _get_src_permutation_idx(indices, device)
        num_targets = max(sum(len(target["symbol_labels"]) for target in targets), 1)
        source_locked = outputs["pred_locked"][
            batch_indices,
            query_indices,
        ].squeeze(-1)
        target_locked = torch.cat(
            [
                targets[batch_index]["locked"].to(device)[target_indices]
                for batch_index, (_, target_indices) in enumerate(indices)
            ]
        )
        loss_locked = (
            F.binary_cross_entropy_with_logits(
                source_locked,
                target_locked,
                reduction="sum",
            )
            / num_targets
            * self.locked_coef
        )
        losses = {
            "loss_class": detector_losses["loss_vfl"],
            "loss_bbox": detector_losses["loss_bbox"],
            "loss_giou": detector_losses["loss_giou"],
            "loss_locked": loss_locked,
            "loss_aux": _sum_detection_auxiliary_losses(
                detector_losses,
                outputs,
            ),
        }
        losses["loss_total"] = sum(losses.values())
        return losses


class TextCriterion(nn.Module):
    """组合文字检测损失和匹配文字query的CTC编号损失。"""

    def __init__(
        self,
        bbox_coef=10.0,
        giou_coef=4.0,
        ctc_coef=1.0,
    ):
        super().__init__()
        self.matcher_impl, self.detector_criterion = _build_detector_criterion(
            1,
            bbox_coef,
            giou_coef,
        )
        self.ctc_coef = ctc_coef
        self.ctc_loss = nn.CTCLoss(
            blank=CTC_BLANK_INDEX,
            reduction="sum",
            zero_infinity=True,
        )

    @staticmethod
    def _prepare_targets(targets, device):
        return [
            {
                "labels": target["text_labels"].to(device),
                "boxes": target["text_boxes"].to(device),
            }
            for target in targets
        ]

    @torch.no_grad()
    def matcher(self, outputs, targets):
        detector_targets = self._prepare_targets(
            targets,
            outputs["pred_logits"].device,
        )
        return self.matcher_impl(outputs, detector_targets)["indices"]

    def forward(self, outputs, targets, indices=None):
        device = outputs["pred_logits"].device
        detector_targets = self._prepare_targets(targets, device)
        detector_losses = self.detector_criterion(outputs, detector_targets)
        if indices is None:
            indices = self.matcher(outputs, targets)
        batch_indices, query_indices = _get_src_permutation_idx(indices, device)
        matched_logits = outputs["pred_identifier_logits"][
            batch_indices,
            query_indices,
        ]
        matched_identifiers = [
            targets[batch_index]["identifiers"][target_index]
            for batch_index, (_, target_indices) in enumerate(indices)
            for target_index in target_indices.tolist()
        ]
        encoded_targets, target_lengths = encode_identifiers(
            matched_identifiers,
            device=device,
        )
        input_lengths = torch.full(
            (len(matched_identifiers),),
            matched_logits.shape[1],
            dtype=torch.long,
            device=device,
        )
        num_targets = max(len(matched_identifiers), 1)
        loss_ctc = (
            self.ctc_loss(
                matched_logits.log_softmax(-1).transpose(0, 1),
                encoded_targets,
                input_lengths,
                target_lengths,
            )
            / num_targets
            * self.ctc_coef
        )
        losses = {
            "loss_class": detector_losses["loss_vfl"],
            "loss_bbox": detector_losses["loss_bbox"],
            "loss_giou": detector_losses["loss_giou"],
            "loss_ctc": loss_ctc,
            "loss_aux": _sum_detection_auxiliary_losses(
                detector_losses,
                outputs,
            ),
        }
        losses["loss_total"] = sum(losses.values())
        return losses


def build_symbol_criterion(
    num_classes=4,
    bbox_coef=5.0,
    giou_coef=2.0,
    locked_coef=1.0,
):
    """构建图元模型损失。"""
    return SymbolCriterion(
        num_classes=num_classes,
        bbox_coef=bbox_coef,
        giou_coef=giou_coef,
        locked_coef=locked_coef,
    )


def build_text_criterion(
    bbox_coef=10.0,
    giou_coef=4.0,
    ctc_coef=1.0,
):
    """构建文字检测识别模型损失。"""
    return TextCriterion(
        bbox_coef=bbox_coef,
        giou_coef=giou_coef,
        ctc_coef=ctc_coef,
    )
