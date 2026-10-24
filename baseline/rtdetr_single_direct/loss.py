"""
RT-DETRv2 实体模型的联合损失。

图元分类、图元框、decoder辅助监督和去噪监督沿用官方 RT-DETRv2；
同一匹配 query 额外计算文字框、锁状态和编号 CTC 损失。
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import generalized_box_iou
from utils import CTC_BLANK_INDEX, cxcywh_to_xyxy, encode_identifiers



PROJECT_ROOT = Path(__file__).resolve().parents[2]
RTDETR_ROOT = PROJECT_ROOT / "third_party" / "rtdetrv2"
if str(RTDETR_ROOT) not in sys.path:
    sys.path.insert(0, str(RTDETR_ROOT))

from src.zoo.rtdetr.matcher import HungarianMatcher  # noqa: E402
from src.zoo.rtdetr.rtdetrv2_criterion import RTDETRCriterionv2  # noqa: E402


class EntityRTDETRCriterion(nn.Module):
    """组合官方 RT-DETRv2 检测损失与实体属性损失。"""

    def __init__(
        self,
        num_classes=4,
        text_bbox_coef=5.0,
        text_giou_coef=2.0,
        locked_coef=1.0,
        ctc_coef=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.text_bbox_coef = text_bbox_coef
        self.text_giou_coef = text_giou_coef
        self.locked_coef = locked_coef
        self.ctc_coef = ctc_coef

        self.detector_matcher = HungarianMatcher(
            weight_dict={"cost_class": 2.0, "cost_bbox": 5.0, "cost_giou": 2.0},
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        )
        self.detector_criterion = RTDETRCriterionv2(
            matcher=self.detector_matcher,
            weight_dict={"loss_vfl": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
            losses=["vfl", "boxes"],
            alpha=0.75,
            gamma=2.0,
            num_classes=num_classes,
        )
        self.ctc_loss = nn.CTCLoss(
            blank=CTC_BLANK_INDEX,
            reduction="sum",
            zero_infinity=True,
        )

        # 返回值已经乘过对应权重，保留该字典以兼容训练脚本的分项统计。
        self.weight_dict = {
            "loss_class": 1.0,
            "loss_symbol_bbox": 1.0,
            "loss_symbol_giou": 1.0,
            "loss_text_bbox": 1.0,
            "loss_text_giou": 1.0,
            "loss_text_aux": 1.0,
            "loss_locked": 1.0,
            "loss_identifier_ctc": 1.0,
        }

    @staticmethod
    def _prepare_detector_targets(targets, device):
        return [
            {
                "labels": target["labels"].to(device),
                "boxes": target["symbol_boxes"].to(device),
            }
            for target in targets
        ]

    @staticmethod
    def _get_src_permutation_idx(indices, device):
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

    @torch.no_grad()
    def matcher(self, outputs, targets):
        """使用官方图元分类与图元框代价完成一对一匹配。"""
        device = outputs["pred_logits"].device
        detector_targets = self._prepare_detector_targets(targets, device)
        matched = self.detector_matcher(
            {
                "pred_logits": outputs["pred_logits"],
                "pred_boxes": outputs["pred_symbol_boxes"],
            },
            detector_targets,
        )
        return matched["indices"]

    def _loss_text_boxes(self, outputs, targets, indices, num_entities):
        device = outputs["pred_text_boxes"].device
        batch_indices, query_indices = self._get_src_permutation_idx(indices, device)
        if len(query_indices) == 0:
            zero = outputs["pred_text_boxes"].sum() * 0
            return zero, zero

        source_boxes = outputs["pred_text_boxes"][batch_indices, query_indices]
        target_boxes = torch.cat(
            [
                targets[batch_index]["text_boxes"].to(device)[target_indices]
                for batch_index, (_, target_indices) in enumerate(indices)
            ]
        )
        loss_bbox = (
            F.l1_loss(source_boxes, target_boxes, reduction="sum")
            / num_entities
            * self.text_bbox_coef
        )
        loss_giou = (
            (
                1
                - torch.diag(
                    generalized_box_iou(
                        cxcywh_to_xyxy(source_boxes),
                        cxcywh_to_xyxy(target_boxes),
                    )
                )
            ).sum()
            / num_entities
            * self.text_giou_coef
        )
        return loss_bbox, loss_giou

    def _loss_locked(self, outputs, targets, indices, num_entities):
        device = outputs["pred_locked"].device
        batch_indices, query_indices = self._get_src_permutation_idx(indices, device)
        if len(query_indices) == 0:
            return outputs["pred_locked"].sum() * 0

        source_locked = outputs["pred_locked"][
            batch_indices, query_indices
        ].squeeze(-1)
        target_locked = torch.cat(
            [
                targets[batch_index]["locked"].to(device)[target_indices]
                for batch_index, (_, target_indices) in enumerate(indices)
            ]
        )
        return (
            F.binary_cross_entropy_with_logits(
                source_locked,
                target_locked,
                reduction="sum",
            )
            / num_entities
            * self.locked_coef
        )

    def _loss_identifiers(self, outputs, targets, indices, num_entities):
        identifier_logits = outputs["pred_identifier_logits"]
        device = identifier_logits.device
        batch_indices, query_indices = self._get_src_permutation_idx(indices, device)
        if len(query_indices) == 0:
            return identifier_logits.sum() * 0

        matched_logits = identifier_logits[batch_indices, query_indices]
        matched_identifiers = [
            targets[batch_index]["identifiers"][target_index]
            for batch_index, (_, target_indices) in enumerate(indices)
            for target_index in target_indices.tolist()
        ]
        encoded_targets, target_lengths = encode_identifiers(
            matched_identifiers,
            device=device,
        )

        sequence_length = matched_logits.shape[1]
        minimum_lengths = [
            len(identifier)
            + sum(
                left == right
                for left, right in zip(identifier, identifier[1:])
            )
            for identifier in matched_identifiers
        ]
        if any(length > sequence_length for length in minimum_lengths):
            raise ValueError("CTC时间步不足以表达当前编号，请增大文字识别序列长度")

        log_probs = matched_logits.log_softmax(-1).transpose(0, 1)
        input_lengths = torch.full(
            (len(matched_identifiers),),
            sequence_length,
            dtype=torch.long,
            device=device,
        )
        return (
            self.ctc_loss(
                log_probs,
                encoded_targets,
                input_lengths,
                target_lengths,
            )
            / num_entities
            * self.ctc_coef
        )

    def forward(self, outputs, targets, indices=None):
        device = outputs["pred_logits"].device
        detector_targets = self._prepare_detector_targets(targets, device)
        detector_losses = self.detector_criterion(outputs, detector_targets)
        if indices is None:
            indices = self.matcher(outputs, targets)

        num_entities = max(sum(len(target["labels"]) for target in targets), 1)
        loss_text_bbox, loss_text_giou = self._loss_text_boxes(
            outputs,
            targets,
            indices,
            num_entities,
        )
        # 中间层使用与最终层相同的匹配结果和框损失，监督逐层文字框修正。
        loss_text_aux = outputs["pred_text_boxes"].sum() * 0
        for auxiliary_outputs in outputs.get("text_aux_outputs", []):
            auxiliary_bbox, auxiliary_giou = self._loss_text_boxes(
                auxiliary_outputs,
                targets,
                indices,
                num_entities,
            )
            loss_text_aux = (
                loss_text_aux
                + auxiliary_bbox
                + auxiliary_giou
            )
        loss_locked = self._loss_locked(outputs, targets, indices, num_entities)
        loss_identifier_ctc = self._loss_identifiers(
            outputs,
            targets,
            indices,
            num_entities,
        )

        primary_detector_names = {"loss_vfl", "loss_bbox", "loss_giou"}
        loss_detection_aux = sum(
            (
                loss
                for name, loss in detector_losses.items()
                if name not in primary_detector_names
            ),
            outputs["pred_logits"].sum() * 0,
        )
        losses = {
            "loss_class": detector_losses["loss_vfl"],
            "loss_symbol_bbox": detector_losses["loss_bbox"],
            "loss_symbol_giou": detector_losses["loss_giou"],
            "loss_text_bbox": loss_text_bbox,
            "loss_text_giou": loss_text_giou,
            "loss_text_aux": loss_text_aux,
            "loss_locked": loss_locked,
            "loss_identifier_ctc": loss_identifier_ctc,
            "loss_detection_aux": loss_detection_aux,
        }
        losses["loss_total"] = sum(losses.values())
        return losses


def build_criterion(
    num_classes=4,
    text_bbox_coef=5.0,
    text_giou_coef=2.0,
    locked_coef=1.0,
    ctc_coef=1.0,
):
    return EntityRTDETRCriterion(
        num_classes=num_classes,
        text_bbox_coef=text_bbox_coef,
        text_giou_coef=text_giou_coef,
        locked_coef=locked_coef,
        ctc_coef=ctc_coef,
    )
