"""
简易Entity-DETR基线使用的工具函数。
"""
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou



CTC_CHARACTERS = "0123456789"
CTC_BLANK_INDEX = len(CTC_CHARACTERS)
CTC_NUM_CLASSES = len(CTC_CHARACTERS) + 1



def _prepare_boxes(boxes):
    """将输入转换为浮点 Tensor，并检查最后一维是否为4。"""
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.ndim == 0 or boxes.shape[-1] != 4:
        raise ValueError("bbox 的 shape 必须为 [4] 或 [..., 4]")
    return boxes


def cxcywh_to_xyxy(boxes):
    """将 `[cx, cy, w, h]` 转换为计算 GIoU 所需的 `[x1, y1, x2, y2]`。"""
    boxes = _prepare_boxes(boxes)
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )


def encode_identifiers(identifiers, device=None):
    """将数字字符串列表编码为 CTCLoss 使用的一维目标和长度。"""
    encoded = []
    lengths = []
    for identifier in identifiers:
        identifier = str(identifier)
        if not identifier or any(character not in CTC_CHARACTERS for character in identifier):
            raise ValueError(f"编号必须是非空纯数字字符串，实际为：{identifier!r}")
        encoded.extend(int(character) for character in identifier)
        lengths.append(len(identifier))

    return (
        torch.tensor(encoded, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def ctc_greedy_decode(identifier_logits, blank_index=CTC_BLANK_INDEX):
    """
    将 `[B, Q, T, C]` 的字符 logits 贪心解码为二维编号字符串列表。

    解码顺序是先合并相邻重复类别，再删除 CTC blank。
    """
    if identifier_logits.ndim != 4:
        raise ValueError("identifier_logits 的 shape 必须为 [B, Q, T, C]")
    if not 0 <= blank_index < identifier_logits.shape[-1]:
        raise ValueError("blank_index 超出字符类别范围")

    predicted_indices = identifier_logits.argmax(dim=-1).detach().cpu()
    decoded_batch = []
    for image_predictions in predicted_indices:
        decoded_queries = []
        for sequence in image_predictions:
            previous_index = None
            characters = []
            for character_index in sequence.tolist():
                if character_index != previous_index and character_index != blank_index:
                    if character_index >= len(CTC_CHARACTERS):
                        raise ValueError("预测字符索引超出当前数字字符集")
                    characters.append(CTC_CHARACTERS[character_index])
                previous_index = character_index
            decoded_queries.append("".join(characters))
        decoded_batch.append(decoded_queries)
    return decoded_batch


def summarize_entity_metrics(
    true_positive,
    false_positive,
    false_negative,
):
    """根据整个验证集累计的实体计数计算准确率、精确率、召回率和F1。"""
    accuracy_denominator = true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    entity_accuracy = (
        true_positive / accuracy_denominator
        if accuracy_denominator > 0 else 0.0
    )
    entity_precision = (
        true_positive / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    entity_recall = (
        true_positive / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    entity_f1 = (
        2 * entity_precision * entity_recall
        / (entity_precision + entity_recall)
        if entity_precision + entity_recall > 0 else 0.0
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "entity_accuracy": entity_accuracy,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
    }


@torch.no_grad()
def calculate_entity_metrics(
    outputs,
    targets,
    entity_conf_threshold=0.5,
    symbol_iou_threshold=0.5,
    text_iou_threshold=0.5,
    lock_threshold=0.5,
):
    """
    对一个batch执行实体级严格评估。

    预测与标签只根据图元框IoU进行一对一匹配；匹配后还必须同时满足
    图元类别、锁状态、文字框IoU和编号完全正确，才计为一个正确实体。
    """
    thresholds = (
        entity_conf_threshold,
        symbol_iou_threshold,
        text_iou_threshold,
        lock_threshold,
    )
    if any(threshold < 0 or threshold > 1 for threshold in thresholds):
        raise ValueError("实体评估的所有阈值都必须位于[0, 1]")

    batch_size, num_queries = outputs["pred_logits"].shape[:2]
    if len(targets) != batch_size:
        raise ValueError("targets数量必须与batch size一致")

    decoded_identifiers = ctc_greedy_decode(
        outputs["pred_identifier_logits"]
    )
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for batch_index in range(batch_size):
        class_probabilities = outputs["pred_logits"][batch_index].softmax(-1)
        # 使用四个前景类别中的最高概率作为实体置信度。
        foreground_scores, foreground_labels = class_probabilities[:, :-1].max(-1)
        keep = foreground_scores >= entity_conf_threshold
        kept_query_indices = torch.arange(
            num_queries,
            device=keep.device,
        )[keep]

        predicted_labels = foreground_labels[keep]
        predicted_locked = (
            outputs["pred_locked"][batch_index, keep].sigmoid().squeeze(-1)
            >= lock_threshold
        )
        predicted_symbol_boxes = cxcywh_to_xyxy(
            outputs["pred_symbol_boxes"][batch_index, keep]
        ).clamp(0, 1)
        predicted_text_boxes = cxcywh_to_xyxy(
            outputs["pred_text_boxes"][batch_index, keep]
        ).clamp(0, 1)
        predicted_identifiers = [
            decoded_identifiers[batch_index][query_index]
            for query_index in kept_query_indices.tolist()
        ]

        target_labels = targets[batch_index]["labels"].to(predicted_labels.device)
        target_locked = targets[batch_index]["locked"].to(predicted_labels.device)
        target_symbol_boxes = cxcywh_to_xyxy(
            targets[batch_index]["symbol_boxes"].to(predicted_labels.device)
        ).clamp(0, 1)
        target_text_boxes = cxcywh_to_xyxy(
            targets[batch_index]["text_boxes"].to(predicted_labels.device)
        ).clamp(0, 1)
        target_identifiers = targets[batch_index]["identifiers"]

        num_predictions = len(predicted_labels)
        num_targets = len(target_labels)
        image_true_positive = 0

        if num_predictions > 0 and num_targets > 0:
            symbol_ious = box_iou(
                predicted_symbol_boxes,
                target_symbol_boxes,
            )
            text_ious = box_iou(
                predicted_text_boxes,
                target_text_boxes,
            )

            # 无效配对使用极大代价，使Hungarian优先保留IoU达标的匹配。
            valid_symbol_pairs = symbol_ious >= symbol_iou_threshold
            matching_cost = torch.where(
                valid_symbol_pairs,
                1 - symbol_ious,
                torch.full_like(symbol_ious, 1e6),
            )
            prediction_indices, target_indices = linear_sum_assignment(
                matching_cost.cpu().numpy()
            )

            for prediction_index, target_index in zip(
                prediction_indices,
                target_indices,
            ):
                if not valid_symbol_pairs[prediction_index, target_index]:
                    continue

                is_exact_entity = (
                    predicted_labels[prediction_index] == target_labels[target_index]
                    and predicted_locked[prediction_index]
                    == target_locked[target_index].bool()
                    and text_ious[prediction_index, target_index]
                    >= text_iou_threshold
                    and predicted_identifiers[prediction_index]
                    == str(target_identifiers[target_index])
                )
                if bool(is_exact_entity):
                    image_true_positive += 1

        # 非完全正确的预测计为FP，未被完整恢复的标签计为FN。
        image_false_positive = num_predictions - image_true_positive
        image_false_negative = num_targets - image_true_positive
        true_positive += image_true_positive
        false_positive += image_false_positive
        false_negative += image_false_negative
    return summarize_entity_metrics(
        true_positive,
        false_positive,
        false_negative,
    )


def entity_collate_fn(batch):
    """堆叠等尺寸图片，同时保留每张图片数量不同的实体标注。"""
    images, targets = zip(*batch)
    image_shapes = {tuple(image.shape) for image in images}
    if len(image_shapes) != 1:
        raise ValueError("同一batch中的图片尺寸不同，请设置统一的image_size")
    return torch.stack(images), list(targets)
