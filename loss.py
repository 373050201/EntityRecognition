"""RT-DETRv2层次化实体模型的图元、文字集合与CTC联合损失。"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops import box_iou, generalized_box_iou
from utils import CTC_BLANK_INDEX, cxcywh_to_xyxy, encode_texts



PROJECT_ROOT = Path(__file__).resolve().parent
RTDETR_ROOT = PROJECT_ROOT / "third_party" / "rtdetrv2"
if str(RTDETR_ROOT) not in sys.path:
    sys.path.insert(0, str(RTDETR_ROOT))

from src.zoo.rtdetr.matcher import HungarianMatcher  # noqa: E402
from src.zoo.rtdetr.rtdetrv2_criterion import RTDETRCriterionv2  # noqa: E402


class EntityRTDETRCriterion(nn.Module):
    """组合外层图元匹配、实体内文字集合匹配和共享CTC损失。"""

    def __init__(
        self,
        num_classes=32,
        text_bbox_coef=5.0,
        text_giou_coef=2.0,
        text_presence_coef=1.0,
        text_duplicate_coef=0.5,
        text_duplicate_iou_threshold=0.5,
        ctc_coef=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.text_bbox_coef = text_bbox_coef
        self.text_giou_coef = text_giou_coef
        self.text_presence_coef = text_presence_coef
        self.text_duplicate_coef = text_duplicate_coef
        self.text_duplicate_iou_threshold = text_duplicate_iou_threshold
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
        self.weight_dict = {
            "loss_class": 1.0,
            "loss_symbol_bbox": 1.0,
            "loss_symbol_giou": 1.0,
            "loss_text_presence": 1.0,
            "loss_text_duplicate": 1.0,
            "loss_text_bbox": 1.0,
            "loss_text_giou": 1.0,
            "loss_text_aux": 1.0,
            "loss_text_ctc": 1.0,
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

    @torch.no_grad()
    def matcher(self, outputs, targets):
        """使用图元分类与图元框代价执行外层实体匹配。"""
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

    @torch.no_grad()
    def match_text_slots(self, outputs, targets, indices):
        """在每个已匹配实体内部用L1和GIoU对齐预测槽位与真实文字集合。"""
        device = outputs["pred_text_boxes"].device
        text_matches = []
        for batch_index, (query_indices, target_indices) in enumerate(indices):
            matched_queries = []
            matched_targets = []
            matched_slots = []
            matched_texts = []
            for query_index, target_index in zip(
                query_indices.tolist(),
                target_indices.tolist(),
            ):
                if not bool(targets[batch_index]["text_supervision"][target_index]):
                    continue
                valid_text_indices = torch.where(
                    targets[batch_index]["text_valid"][target_index]
                )[0]
                if len(valid_text_indices) == 0:
                    continue
                source_boxes = outputs["pred_text_boxes"][
                    batch_index,
                    query_index,
                ]
                target_boxes = targets[batch_index]["text_boxes"][
                    target_index,
                    valid_text_indices,
                ].to(device)
                bbox_cost = torch.cdist(source_boxes, target_boxes, p=1)
                giou_cost = -generalized_box_iou(
                    cxcywh_to_xyxy(source_boxes),
                    cxcywh_to_xyxy(target_boxes),
                )
                cost = (
                    self.text_bbox_coef * bbox_cost
                    + self.text_giou_coef * giou_cost
                )
                slot_indices, local_text_indices = linear_sum_assignment(
                    cost.detach().cpu().numpy()
                )
                for slot_index, local_text_index in zip(
                    slot_indices.tolist(),
                    local_text_indices.tolist(),
                ):
                    matched_queries.append(query_index)
                    matched_targets.append(target_index)
                    matched_slots.append(slot_index)
                    matched_texts.append(
                        int(valid_text_indices[local_text_index])
                    )
            text_matches.append(
                {
                    "query_indices": torch.tensor(
                        matched_queries,
                        dtype=torch.long,
                        device=device,
                    ),
                    "target_indices": torch.tensor(
                        matched_targets,
                        dtype=torch.long,
                        device=device,
                    ),
                    "slot_indices": torch.tensor(
                        matched_slots,
                        dtype=torch.long,
                        device=device,
                    ),
                    "text_indices": torch.tensor(
                        matched_texts,
                        dtype=torch.long,
                        device=device,
                    ),
                }
            )
        return text_matches

    def _loss_text_presence(self, outputs, targets, indices, text_matches):
        logits = outputs["pred_text_presence"]
        device = logits.device
        loss = logits.sum() * 0
        supervised_slots = 0
        for batch_index, (query_indices, target_indices) in enumerate(indices):
            if len(query_indices) == 0:
                continue
            supervision_mask = targets[batch_index]["text_supervision"][
                target_indices.cpu()
            ].to(device)
            if not supervision_mask.any():
                continue
            query_indices = query_indices.to(device)[supervision_mask]
            target_indices = target_indices.to(device)[supervision_mask]
            source_logits = logits[batch_index, query_indices]
            target_presence = torch.zeros_like(source_logits)
            matches = text_matches[batch_index]
            for row_index, query_index in enumerate(query_indices.tolist()):
                query_mask = matches["query_indices"] == query_index
                target_presence[
                    row_index,
                    matches["slot_indices"][query_mask],
                ] = 1.0
            loss = loss + F.binary_cross_entropy_with_logits(
                source_logits,
                target_presence,
                reduction="sum",
            )
            supervised_slots += source_logits.numel()
        return loss / max(supervised_slots, 1) * self.text_presence_coef

    def _loss_text_boxes(self, outputs, targets, text_matches):
        device = outputs["pred_text_boxes"].device
        source_box_groups = []
        target_box_groups = []
        for batch_index, matches in enumerate(text_matches):
            if len(matches["slot_indices"]) == 0:
                continue
            source_box_groups.append(
                outputs["pred_text_boxes"][
                    batch_index,
                    matches["query_indices"],
                    matches["slot_indices"],
                ]
            )
            target_box_groups.append(
                targets[batch_index]["text_boxes"].to(device)[
                    matches["target_indices"],
                    matches["text_indices"],
                ]
            )
        if not source_box_groups:
            zero = outputs["pred_text_boxes"].sum() * 0
            return zero, zero

        source_boxes = torch.cat(source_box_groups)
        target_boxes = torch.cat(target_box_groups)
        num_texts = len(source_boxes)
        loss_bbox = (
            F.l1_loss(source_boxes, target_boxes, reduction="sum")
            / num_texts
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
            / num_texts
            * self.text_giou_coef
        )
        return loss_bbox, loss_giou

    def _loss_text_duplicates(self, outputs, text_matches):
        """用重复框IoU加权未匹配槽位的有效概率，且不更新文字框。"""
        logits = outputs["pred_text_presence"]
        text_boxes = outputs["pred_text_boxes"]
        loss = logits.sum() * 0
        unmatched_slot_count = 0
        num_text_slots = logits.shape[-1]

        for batch_index, matches in enumerate(text_matches):
            for query_index in matches["query_indices"].unique().tolist():
                query_mask = matches["query_indices"] == query_index
                matched_slots = matches["slot_indices"][query_mask]
                unmatched_mask = torch.ones(
                    num_text_slots,
                    dtype=torch.bool,
                    device=logits.device,
                )
                unmatched_mask[matched_slots] = False
                unmatched_slots = torch.where(unmatched_mask)[0]
                if len(unmatched_slots) == 0:
                    continue

                # IoU只提供重复程度，不允许该损失通过移动文字框来下降。
                with torch.no_grad():
                    query_boxes = cxcywh_to_xyxy(
                        text_boxes[batch_index, query_index]
                    ).float()
                    duplicate_ious = box_iou(
                        query_boxes[unmatched_slots],
                        query_boxes[matched_slots],
                    ).amax(dim=1)
                    duplicate_weights = (
                        duplicate_ious - self.text_duplicate_iou_threshold
                    ).clamp(min=0)

                unmatched_probabilities = logits[
                    batch_index,
                    query_index,
                    unmatched_slots,
                ].sigmoid()
                loss = loss + (
                    unmatched_probabilities * duplicate_weights
                ).sum()
                unmatched_slot_count += len(unmatched_slots)

        return (
            loss
            / max(unmatched_slot_count, 1)
            * self.text_duplicate_coef
        )

    def _loss_texts(self, outputs, targets, text_matches):
        text_logits = outputs.get("pred_text_logits")
        selected_logits = outputs.get("matched_text_logits")
        matched_logits = []
        matched_texts = []
        selected_index = 0
        for batch_index, matches in enumerate(text_matches):
            for query_index, target_index, slot_index, text_index in zip(
                matches["query_indices"].tolist(),
                matches["target_indices"].tolist(),
                matches["slot_indices"].tolist(),
                matches["text_indices"].tolist(),
            ):
                if selected_logits is None:
                    matched_logits.append(
                        text_logits[batch_index, query_index, slot_index]
                    )
                else:
                    matched_logits.append(selected_logits[selected_index])
                    selected_index += 1
                matched_texts.append(
                    targets[batch_index]["texts"][target_index][text_index]
                )
        if not matched_logits:
            reference_logits = (
                selected_logits
                if selected_logits is not None
                else text_logits
            )
            return reference_logits.sum() * 0

        if selected_logits is not None and selected_index != len(selected_logits):
            raise ValueError("匹配OCR输出数量与文字匹配数量不一致")

        matched_logits = torch.stack(matched_logits)
        encoded_targets, target_lengths = encode_texts(
            matched_texts,
            device=matched_logits.device,
        )
        sequence_length = matched_logits.shape[1]
        minimum_lengths = [
            len(text) + sum(left == right for left, right in zip(text, text[1:]))
            for text in matched_texts
        ]
        if any(length > sequence_length for length in minimum_lengths):
            raise ValueError("CTC时间步不足以表达当前文字，请增大文字识别序列长度")

        # CTC在FP32下计算，避免混合精度时出现数值下溢。
        log_probs = matched_logits.float().log_softmax(-1).transpose(0, 1)
        input_lengths = torch.full(
            (len(matched_texts),),
            sequence_length,
            dtype=torch.long,
            device=matched_logits.device,
        )
        return (
            self.ctc_loss(
                log_probs,
                encoded_targets,
                input_lengths,
                target_lengths,
            )
            / len(matched_texts)
            * self.ctc_coef
        )

    def forward(self, outputs, targets, indices=None, text_matches=None):
        device = outputs["pred_logits"].device
        detector_targets = self._prepare_detector_targets(targets, device)
        detector_losses = self.detector_criterion(outputs, detector_targets)
        if indices is None:
            indices = self.matcher(outputs, targets)
        if text_matches is None:
            text_matches = self.match_text_slots(outputs, targets, indices)

        loss_text_presence = self._loss_text_presence(
            outputs,
            targets,
            indices,
            text_matches,
        )
        loss_text_duplicate = self._loss_text_duplicates(
            outputs,
            text_matches,
        )
        loss_text_bbox, loss_text_giou = self._loss_text_boxes(
            outputs,
            targets,
            text_matches,
        )
        loss_text_aux = outputs["pred_text_boxes"].sum() * 0
        for auxiliary_outputs in outputs.get("text_aux_outputs", []):
            auxiliary_bbox, auxiliary_giou = self._loss_text_boxes(
                auxiliary_outputs,
                targets,
                text_matches,
            )
            loss_text_aux = loss_text_aux + auxiliary_bbox + auxiliary_giou
        loss_text_ctc = self._loss_texts(outputs, targets, text_matches)

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
            "loss_text_presence": loss_text_presence,
            "loss_text_duplicate": loss_text_duplicate,
            "loss_text_bbox": loss_text_bbox,
            "loss_text_giou": loss_text_giou,
            "loss_text_aux": loss_text_aux,
            "loss_text_ctc": loss_text_ctc,
            "loss_detection_aux": loss_detection_aux,
        }
        losses["loss_total"] = sum(losses.values())
        return losses


def build_criterion(
    num_classes=32,
    text_bbox_coef=5.0,
    text_giou_coef=2.0,
    text_presence_coef=1.0,
    text_duplicate_coef=0.5,
    text_duplicate_iou_threshold=0.5,
    ctc_coef=1.0,
):
    return EntityRTDETRCriterion(
        num_classes=num_classes,
        text_bbox_coef=text_bbox_coef,
        text_giou_coef=text_giou_coef,
        text_presence_coef=text_presence_coef,
        text_duplicate_coef=text_duplicate_coef,
        text_duplicate_iou_threshold=text_duplicate_iou_threshold,
        ctc_coef=ctc_coef,
    )
