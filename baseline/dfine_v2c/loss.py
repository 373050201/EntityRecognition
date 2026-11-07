"""D-FINE-V2C 的图元检测、文字框、锁状态和 CTC 联合损失。"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import generalized_box_iou


BASELINE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_ROOT.parents[1]
DFINE_ROOT = PROJECT_ROOT / "third_party" / "dfine"
for import_root in (PROJECT_ROOT, DFINE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from utils import CTC_BLANK_INDEX, cxcywh_to_xyxy, encode_identifiers  # noqa: E402
from src.zoo.dfine.dfine_criterion import DFINECriterion  # noqa: E402
from src.zoo.dfine.matcher import HungarianMatcher  # noqa: E402


class EntityDFINECriterion(nn.Module):
    """保留 D-FINE 原生检测损失，并增加 V2-C 实体属性损失。"""

    def __init__(
        self,
        num_classes=4,
        text_bbox_coef=10.0,
        text_giou_coef=4.0,
        locked_coef=1.0,
        ctc_coef=1.0,
        text_fgl_coef=0.15,
        text_ddf_coef=1.5,
    ):
        super().__init__()
        self.text_bbox_coef = text_bbox_coef
        self.text_giou_coef = text_giou_coef
        self.locked_coef = locked_coef
        self.ctc_coef = ctc_coef
        self.text_fgl_coef = text_fgl_coef
        self.text_ddf_coef = text_ddf_coef
        self.detector_matcher = HungarianMatcher(
            weight_dict={
                "cost_class": 2.0,
                "cost_bbox": 5.0,
                "cost_giou": 2.0,
            },
            use_focal_loss=True,
            alpha=0.25,
            gamma=2.0,
        )
        self.detector_criterion = DFINECriterion(
            matcher=self.detector_matcher,
            weight_dict={
                "loss_vfl": 1.0,
                "loss_bbox": 5.0,
                "loss_giou": 2.0,
                "loss_fgl": 0.15,
                "loss_ddf": 1.5,
            },
            losses=["vfl", "boxes", "local"],
            alpha=0.75,
            gamma=2.0,
            num_classes=num_classes,
            reg_max=32,
        )
        # 文字FDR拥有独立缓存，避免与图元参考框的FGL目标相互覆盖。
        self.text_fdr_criterion = DFINECriterion(
            matcher=self.detector_matcher,
            weight_dict={"loss_fgl": 1.0, "loss_ddf": 1.0},
            losses=["local"],
            alpha=0.75,
            gamma=2.0,
            num_classes=num_classes,
            reg_max=32,
        )
        self.ctc_loss = nn.CTCLoss(
            blank=CTC_BLANK_INDEX,
            reduction="sum",
            zero_infinity=True,
        )
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
        device = outputs["pred_logits"].device
        detector_targets = self._prepare_detector_targets(targets, device)
        return self.detector_matcher(
            {
                "pred_logits": outputs["pred_logits"],
                "pred_boxes": outputs["pred_symbol_boxes"],
            },
            detector_targets,
        )["indices"]

    def _loss_text_boxes(self, outputs, targets, indices, num_entities):
        device = outputs["pred_text_boxes"].device
        batch_indices, query_indices = self._get_src_permutation_idx(
            indices,
            device,
        )
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
        batch_indices, query_indices = self._get_src_permutation_idx(
            indices,
            device,
        )
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
        batch_indices, query_indices = self._get_src_permutation_idx(
            indices,
            device,
        )
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

    def _detector_losses(self, outputs, targets, indices):
        """训练时使用完整 GO-LSD；评估时只计算最终层可定义的损失。"""
        if "pred_corners" in outputs and "aux_outputs" in outputs:
            return self.detector_criterion(outputs, targets)

        num_entities = max(sum(len(target["labels"]) for target in targets), 1)
        losses = self.detector_criterion.loss_labels_vfl(
            outputs,
            targets,
            indices,
            num_entities,
        )
        losses.update(
            self.detector_criterion.loss_boxes(
                outputs,
                targets,
                indices,
                num_entities,
            )
        )
        weight_dict = self.detector_criterion.weight_dict
        return {
            name: loss * weight_dict[name]
            for name, loss in losses.items()
        }

    def _loss_text_fdr(self, outputs, targets, indices, num_entities):
        """计算文字最终层FGL和浅层DDF自蒸馏损失。"""
        zero = outputs["pred_text_boxes"].sum() * 0
        if "pred_text_corners" not in outputs:
            return zero, zero

        device = outputs["pred_text_boxes"].device
        text_targets = [
            {
                "labels": target["labels"].to(device),
                "boxes": target["text_boxes"].to(device),
            }
            for target in targets
        ]
        self.text_fdr_criterion._clear_cache()
        final_fdr_outputs = {
            "pred_boxes": outputs["pred_text_boxes"],
            "pred_corners": outputs["pred_text_corners"],
            "ref_points": outputs["text_ref_points"],
            "reg_scale": outputs["reg_scale"],
            "up": outputs["up"],
        }
        final_losses = self.text_fdr_criterion.loss_local(
            final_fdr_outputs,
            text_targets,
            indices,
            num_entities,
        )
        final_fgl = (
            final_losses.get("loss_fgl", zero) * self.text_fgl_coef
        )

        auxiliary_fdr = zero
        for auxiliary_outputs in outputs.get("text_aux_outputs", []):
            auxiliary_fdr_outputs = {
                "pred_boxes": auxiliary_outputs["pred_text_boxes"],
                "pred_corners": auxiliary_outputs["pred_text_corners"],
                "ref_points": auxiliary_outputs["text_ref_points"],
                "teacher_corners": auxiliary_outputs[
                    "teacher_text_corners"
                ],
                "teacher_logits": auxiliary_outputs["teacher_logits"],
                "reg_scale": outputs["reg_scale"],
                "up": outputs["up"],
            }
            auxiliary_losses = self.text_fdr_criterion.loss_local(
                auxiliary_fdr_outputs,
                text_targets,
                indices,
                num_entities,
            )
            auxiliary_fdr = (
                auxiliary_fdr
                + auxiliary_losses.get("loss_fgl", zero)
                * self.text_fgl_coef
                + auxiliary_losses.get("loss_ddf", zero)
                * self.text_ddf_coef
            )
        return final_fgl, auxiliary_fdr

    def forward(self, outputs, targets, indices=None):
        device = outputs["pred_logits"].device
        detector_targets = self._prepare_detector_targets(targets, device)
        if indices is None:
            indices = self.matcher(outputs, targets)
        detector_losses = self._detector_losses(
            outputs,
            detector_targets,
            indices,
        )

        num_entities = max(sum(len(target["labels"]) for target in targets), 1)
        loss_text_bbox, loss_text_giou = self._loss_text_boxes(
            outputs,
            targets,
            indices,
            num_entities,
        )
        loss_text_fgl, loss_text_fdr_aux = self._loss_text_fdr(
            outputs,
            targets,
            indices,
            num_entities,
        )
        loss_text_aux = outputs["pred_text_boxes"].sum() * 0
        for auxiliary_outputs in outputs.get("text_aux_outputs", []):
            auxiliary_bbox, auxiliary_giou = self._loss_text_boxes(
                auxiliary_outputs,
                targets,
                indices,
                num_entities,
            )
            loss_text_aux = loss_text_aux + auxiliary_bbox + auxiliary_giou
        if "text_pre_output" in outputs:
            pre_bbox, pre_giou = self._loss_text_boxes(
                outputs["text_pre_output"],
                targets,
                indices,
                num_entities,
            )
            loss_text_aux = loss_text_aux + pre_bbox + pre_giou
        loss_text_aux = loss_text_aux + loss_text_fdr_aux

        loss_locked = self._loss_locked(
            outputs,
            targets,
            indices,
            num_entities,
        )
        loss_identifier_ctc = self._loss_identifiers(
            outputs,
            targets,
            indices,
            num_entities,
        )

        zero = outputs["pred_logits"].sum() * 0
        primary_names = {
            "loss_vfl",
            "loss_bbox",
            "loss_giou",
            "loss_fgl",
            "loss_ddf",
        }
        loss_detection_aux = sum(
            (
                value
                for name, value in detector_losses.items()
                if name not in primary_names
            ),
            zero,
        )
        losses = {
            "loss_class": detector_losses.get("loss_vfl", zero),
            "loss_symbol_bbox": (
                detector_losses.get("loss_bbox", zero)
                + detector_losses.get("loss_fgl", zero)
                + detector_losses.get("loss_ddf", zero)
            ),
            "loss_symbol_giou": detector_losses.get("loss_giou", zero),
            "loss_text_bbox": loss_text_bbox + loss_text_fgl,
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
    text_bbox_coef=10.0,
    text_giou_coef=4.0,
    locked_coef=1.0,
    ctc_coef=1.0,
    text_fgl_coef=0.15,
    text_ddf_coef=1.5,
):
    return EntityDFINECriterion(
        num_classes=num_classes,
        text_bbox_coef=text_bbox_coef,
        text_giou_coef=text_giou_coef,
        locked_coef=locked_coef,
        ctc_coef=ctc_coef,
        text_fgl_coef=text_fgl_coef,
        text_ddf_coef=text_ddf_coef,
    )
