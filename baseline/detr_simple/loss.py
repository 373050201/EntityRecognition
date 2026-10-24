"""
Entity-DETR 的实体匹配与损失函数。

同一个 query 同时负责图元类别、图元框、文字框、锁状态和编号。训练时先通过
Hungarian matching 将 query 与真实实体一一匹配，再计算各预测头和 CTC 损失。
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.ops import generalized_box_iou
from .utils import CTC_BLANK_INDEX, cxcywh_to_xyxy, encode_identifiers



class HungarianMatcher(nn.Module):
    """根据类别、双框和锁状态，为每张图片执行实体级一对一匹配。"""

    def __init__(
        self,
        cost_class=2.0,
        cost_symbol_bbox=5.0,
        cost_symbol_giou=2.0,
        cost_text_bbox=5.0,
        cost_text_giou=2.0,
        cost_locked=0.5,
    ):
        super().__init__()
        costs = (
            cost_class,
            cost_symbol_bbox,
            cost_symbol_giou,
            cost_text_bbox,
            cost_text_giou,
            cost_locked,
        )
        if any(cost < 0 for cost in costs):
            raise ValueError("Hungarian matching 的各项代价权重不能为负数")
        if all(cost == 0 for cost in costs):
            raise ValueError("Hungarian matching 至少需要一个非零代价")

        self.cost_class = cost_class
        self.cost_symbol_bbox = cost_symbol_bbox
        self.cost_symbol_giou = cost_symbol_giou
        self.cost_text_bbox = cost_text_bbox
        self.cost_text_giou = cost_text_giou
        self.cost_locked = cost_locked

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: 模型输出，query维度为 `[B, Q, ...]`。
            targets: 长度为B的列表，每项包含当前图片的N个真实实体。

        Returns:
            长度为B的列表，每项是 `(query_indices, target_indices)`。
        """
        batch_size, num_queries = outputs["pred_logits"].shape[:2]
        if len(targets) != batch_size:
            raise ValueError("targets数量必须与batch size一致")

        device = outputs["pred_logits"].device
        indices = []
        for batch_index in range(batch_size):
            target_labels = targets[batch_index]["labels"].to(device)
            num_targets = len(target_labels)

            if num_targets == 0:
                empty = torch.empty(0, dtype=torch.long, device=device)
                indices.append((empty, empty))
                continue

            # [Q, 5] -> [Q, N]，真实类别概率越高，分类代价越小。
            class_prob = outputs["pred_logits"][batch_index].softmax(-1)
            cost_class = -class_prob[:, target_labels]

            predicted_symbol_boxes = outputs["pred_symbol_boxes"][batch_index]
            target_symbol_boxes = targets[batch_index]["symbol_boxes"].to(device)
            cost_symbol_bbox = torch.cdist(
                predicted_symbol_boxes,
                target_symbol_boxes,
                p=1,
            )
            cost_symbol_giou = -generalized_box_iou(
                cxcywh_to_xyxy(predicted_symbol_boxes),
                cxcywh_to_xyxy(target_symbol_boxes),
            )

            predicted_text_boxes = outputs["pred_text_boxes"][batch_index]
            target_text_boxes = targets[batch_index]["text_boxes"].to(device)
            cost_text_bbox = torch.cdist(
                predicted_text_boxes,
                target_text_boxes,
                p=1,
            )
            cost_text_giou = -generalized_box_iou(
                cxcywh_to_xyxy(predicted_text_boxes),
                cxcywh_to_xyxy(target_text_boxes),
            )

            predicted_locked = outputs["pred_locked"][batch_index].squeeze(-1)
            target_locked = targets[batch_index]["locked"].to(device)
            cost_locked = F.binary_cross_entropy_with_logits(
                predicted_locked[:, None].expand(num_queries, num_targets),
                target_locked[None, :].expand(num_queries, num_targets),
                reduction="none",
            )

            # 总代价矩阵为[Q,N]，一次匹配绑定同一实体的全部预测结果。
            total_cost = (
                self.cost_class * cost_class
                + self.cost_symbol_bbox * cost_symbol_bbox
                + self.cost_symbol_giou * cost_symbol_giou
                + self.cost_text_bbox * cost_text_bbox
                + self.cost_text_giou * cost_text_giou
                + self.cost_locked * cost_locked
            )
            query_indices, target_indices = linear_sum_assignment(
                total_cost.detach().cpu().numpy()
            )
            indices.append(
                (
                    torch.as_tensor(query_indices, dtype=torch.long, device=device),
                    torch.as_tensor(target_indices, dtype=torch.long, device=device),
                )
            )

        return indices # 长度为B的list，每个元素为(query_indices, target_indices)，即图i中所有查询idx和实体idx的对应


class SetCriterion(nn.Module):
    """计算 Entity-DETR 的分类、双框、锁状态和编号识别损失。"""

    def __init__(
        self,
        num_classes=4,
        matcher=None,
        eos_coef=0.5,
        weight_dict=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher if matcher is not None else HungarianMatcher()
        self.weight_dict = weight_dict or {
            "loss_class": 1.0,
            "loss_symbol_bbox": 5.0,
            "loss_symbol_giou": 2.0,
            "loss_text_bbox": 5.0,
            "loss_text_giou": 2.0,
            "loss_locked": 1.0,
            "loss_identifier_ctc": 1.0,
        }
        self.ctc_loss = nn.CTCLoss(
            blank=CTC_BLANK_INDEX,
            reduction="sum",
            zero_infinity=True,
        )

        # 最后一类是no-object，降低其权重以缓解大量空query的影响。
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    @staticmethod
    def _get_src_permutation_idx(indices):
        """把每张图片的匹配query索引拼成batch高级索引。"""
        batch_indices = torch.cat(
            [
                torch.full_like(query_indices, batch_index)
                for batch_index, (query_indices, _) in enumerate(indices)
            ]
        )
        query_indices = torch.cat(
            [query_indices for query_indices, _ in indices]
        )
        return batch_indices, query_indices # 均为[N]，N为所有图片所有已匹配实体的查询总数，分别为图片idx，已匹配的查询idx

    def loss_labels(self, outputs, targets, indices):
        """对全部query计算四分类加no-object的交叉熵损失。"""
        source_logits = outputs["pred_logits"]  # [B, Q, 5]
        target_classes = torch.full(
            source_logits.shape[:2],
            self.num_classes,
            dtype=torch.long,
            device=source_logits.device,
        ) # 构建GT，[B, Q]，全部元素初始化为4（no_object）

        batch_indices, query_indices = self._get_src_permutation_idx(indices)
        if len(query_indices) > 0:
            matched_target_classes = torch.cat(
                [
                    targets[batch_index]["labels"].to(source_logits.device)[
                        target_indices
                    ]
                    for batch_index, (_, target_indices) in enumerate(indices)
                ]
            ) # 用target_indices取出真实类别
            target_classes[batch_indices, query_indices] = matched_target_classes # 用真实类别替换对应位置的4

        loss_class = F.cross_entropy(
            source_logits.transpose(1, 2),
            target_classes,
            weight=self.empty_weight.to(source_logits.device),
        )
        return {"loss_class": loss_class}

    def loss_boxes(self, outputs, targets, indices, num_entities):
        """对匹配query计算图元框和文字框的L1与GIoU损失。"""
        batch_indices, query_indices = self._get_src_permutation_idx(indices)
        if len(query_indices) == 0:
            zero = outputs["pred_symbol_boxes"].sum() * 0
            return {
                "loss_symbol_bbox": zero,
                "loss_symbol_giou": zero,
                "loss_text_bbox": zero,
                "loss_text_giou": zero,
            }

        source_symbol_boxes = outputs["pred_symbol_boxes"][
            batch_indices, query_indices
        ] # [M, 4]，模型预测的图元框，M为所有图片所有匹配查询的总数
        source_text_boxes = outputs["pred_text_boxes"][
            batch_indices, query_indices
        ] # [M, 4]，模型预测的文字框
        target_symbol_boxes = torch.cat(
            [
                targets[batch_index]["symbol_boxes"].to(
                    source_symbol_boxes.device
                )[target_indices]
                for batch_index, (_, target_indices) in enumerate(indices)
            ]
        ) # [M, 4]，对应的真实图元框
        target_text_boxes = torch.cat(
            [
                targets[batch_index]["text_boxes"].to(source_text_boxes.device)[
                    target_indices
                ]
                for batch_index, (_, target_indices) in enumerate(indices)
            ]
        )# [M, 4]，对应的真实文字框

        # 分别计算L1损失
        loss_symbol_bbox = F.l1_loss(
            source_symbol_boxes,
            target_symbol_boxes,
            reduction="sum",
        ) / num_entities
        loss_text_bbox = F.l1_loss(
            source_text_boxes,
            target_text_boxes,
            reduction="sum",
        ) / num_entities

        # 分别计算GIoU损失
        symbol_giou = generalized_box_iou(
            cxcywh_to_xyxy(source_symbol_boxes),
            cxcywh_to_xyxy(target_symbol_boxes),
        )
        text_giou = generalized_box_iou(
            cxcywh_to_xyxy(source_text_boxes),
            cxcywh_to_xyxy(target_text_boxes),
        )
        loss_symbol_giou = (1 - torch.diag(symbol_giou)).sum() / num_entities # 第i个预测框和第i个真实框对应（对角）
        loss_text_giou = (1 - torch.diag(text_giou)).sum() / num_entities

        return {
            "loss_symbol_bbox": loss_symbol_bbox,
            "loss_symbol_giou": loss_symbol_giou,
            "loss_text_bbox": loss_text_bbox,
            "loss_text_giou": loss_text_giou,
        }

    def loss_locked(self, outputs, targets, indices, num_entities):
        """只对匹配query计算锁状态二分类损失。"""
        batch_indices, query_indices = self._get_src_permutation_idx(indices)
        if len(query_indices) == 0:
            return {"loss_locked": outputs["pred_locked"].sum() * 0}

        source_locked = outputs["pred_locked"][
            batch_indices, query_indices
        ].squeeze(-1) # [M, 1]->[M]，模型预测的是否加锁的logits
        target_locked = torch.cat(
            [
                targets[batch_index]["locked"].to(source_locked.device)[
                    target_indices
                ]
                for batch_index, (_, target_indices) in enumerate(indices)
            ]
        ) # [M]，真实加锁状态，加锁为1
        # 计算BCE损失，自动sigmoid
        loss_locked = F.binary_cross_entropy_with_logits(
            source_locked,
            target_locked,
            reduction="sum",
        ) / num_entities
        return {"loss_locked": loss_locked}

    def loss_identifiers(self, outputs, targets, indices, num_entities):
        """只对匹配query计算编号的CTC序列识别损失。"""
        identifier_logits = outputs["pred_identifier_logits"] # [B, Q, T, 11]
        batch_indices, query_indices = self._get_src_permutation_idx(indices)
        if len(query_indices) == 0:
            return {"loss_identifier_ctc": identifier_logits.sum() * 0}

        matched_logits = identifier_logits[batch_indices, query_indices]
         # [B, Q, T, 11] -> [M, T, 11]，M为所有图片的匹配查询总数。
        matched_identifiers = [
            targets[batch_index]["identifiers"][target_index]
            for batch_index, (_, target_indices) in enumerate(indices)
            for target_index in target_indices.tolist()
        ] # 取出对应实体的真实标识号
        encoded_targets, target_lengths = encode_identifiers(
            matched_identifiers,
            device=matched_logits.device,
        ) # 函数来自utils.py，将字符串式的标识号转换为计算CTC_loss所需的tensor，并返回标识号长度

        # 安全检查，确保模型的时间步数足以给真实标识号编码
        sequence_length = matched_logits.shape[1] # T
        minimum_lengths = [
            len(identifier)
            + sum(
                left == right
                for left, right in zip(identifier, identifier[1:])
            )
            for identifier in matched_identifiers
        ]
        if any(length > sequence_length for length in minimum_lengths):
            raise ValueError(
                "CTC时间步不足以表达当前编号，请增大文字识别序列长度"
            )

        # CTCLoss要求输入为[T, M, C]，并使用log softmax后的对数概率。
        log_probs = matched_logits.log_softmax(-1).transpose(0, 1)
        input_lengths = torch.full( # 有效长度设为T
            (len(matched_identifiers),),
            sequence_length,
            dtype=torch.long,
            device=matched_logits.device,
        )
        loss_identifier_ctc = self.ctc_loss(
            log_probs,
            encoded_targets,
            input_lengths,
            target_lengths,
        ) / num_entities
        return {"loss_identifier_ctc": loss_identifier_ctc}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets):
        """统计预测实体数量误差，仅用于日志观察，不加入总损失。"""
        predicted_classes = outputs["pred_logits"].argmax(-1)
        predicted_counts = (predicted_classes != self.num_classes).sum(1).float()
        target_counts = torch.as_tensor(
            [len(target["labels"]) for target in targets],
            dtype=torch.float32,
            device=predicted_counts.device,
        )
        return {
            "cardinality_error": F.l1_loss(predicted_counts, target_counts) # 预测实体数和真实数的L1损失
        }

    def forward(self, outputs, targets, indices=None):
        # indices中每个真实实体只会匹配一个query，且每个query最多匹配一个实体。
        if indices is None:
            indices = self.matcher(outputs, targets)
        num_entities = max(
            sum(len(target["labels"]) for target in targets),
            1,
        )

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_boxes(outputs, targets, indices, num_entities))
        losses.update(self.loss_locked(outputs, targets, indices, num_entities))
        losses.update(
            self.loss_identifiers(outputs, targets, indices, num_entities)
        )
        losses.update(self.loss_cardinality(outputs, targets))

        # cardinality_error只用于观察，不出现在weight_dict中，也不参与反向传播。
        losses["loss_total"] = sum(
            losses[loss_name] * weight
            for loss_name, weight in self.weight_dict.items()
        )
        return losses


def build_criterion(eos_coef=0.5):
    """按照当前模型头和初始权重构建实体匹配器与损失函数。"""
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_symbol_bbox=5.0,
        cost_symbol_giou=2.0,
        cost_text_bbox=5.0,
        cost_text_giou=2.0,
        cost_locked=0.5,
    )
    return SetCriterion(
        num_classes=4,
        matcher=matcher,
        eos_coef=eos_coef,
    )
