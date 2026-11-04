"""工业流水线的坐标转换、严格实体评估与复现工具。"""
import random
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou



def set_random_seed(seed):
    """固定常用随机源，并保留当前项目使用的高性能CUDA路径。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)


def cxcywh_to_xyxy(boxes):
    """把归一化cxcywh转换为xyxy。"""
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


def summarize_counts(true_positive, false_positive, false_negative):
    """根据全数据集TP、FP、FN计算Accuracy、Precision、Recall和F1。"""
    accuracy_denominator = true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    accuracy = true_positive / accuracy_denominator if accuracy_denominator else 0.0
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def create_entity_breakdown_counts():
    """创建完整实体拆解指标的累计计数器。"""
    return {
        "num_targets": 0,
        "num_characters": 0,
        "symbol_iou_sum": 0.0,
        "text_iou_sum": 0.0,
        "symbol_box_correct": 0,
        "text_box_correct": 0,
        "symbol_class_correct": 0,
        "locked_correct": 0,
        "identifier_character_correct": 0,
        "identifier_exact_correct": 0,
    }


def summarize_entity_breakdown(counts):
    """把完整实体拆解计数转换为平均IoU和准确率。"""
    num_targets = counts["num_targets"]
    num_characters = counts["num_characters"]

    def entity_rate(name):
        return counts[name] / num_targets if num_targets else 0.0

    return {
        "symbol_mean_iou": entity_rate("symbol_iou_sum"),
        "symbol_box_accuracy": entity_rate("symbol_box_correct"),
        "text_mean_iou": entity_rate("text_iou_sum"),
        "text_box_accuracy": entity_rate("text_box_correct"),
        "symbol_class_accuracy": entity_rate("symbol_class_correct"),
        "locked_accuracy": entity_rate("locked_correct"),
        "identifier_character_accuracy": (
            counts["identifier_character_correct"] / num_characters
            if num_characters else 0.0
        ),
        "identifier_exact_accuracy": entity_rate("identifier_exact_correct"),
    }


def _levenshtein_distance(prediction, target):
    """计算两个编号字符串之间的编辑距离。"""
    previous_row = list(range(len(target) + 1))
    for prediction_index, prediction_character in enumerate(prediction, start=1):
        current_row = [prediction_index]
        for target_index, target_character in enumerate(target, start=1):
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[target_index] + 1,
                    previous_row[target_index - 1]
                    + (prediction_character != target_character),
                )
            )
        previous_row = current_row
    return previous_row[-1]


def _match_by_iou(predicted_boxes, target_boxes, iou_threshold):
    """按照IoU执行一对一匹配，并返回预测索引、目标索引和IoU矩阵。"""
    if len(predicted_boxes) == 0 or len(target_boxes) == 0:
        return [], [], torch.empty(
            (len(predicted_boxes), len(target_boxes)),
            device=predicted_boxes.device,
        )
    ious = box_iou(predicted_boxes, target_boxes)
    valid_pairs = ious >= iou_threshold
    cost = torch.where(valid_pairs, 1 - ious, 1e3 + (1 - ious))
    prediction_indices, target_indices = linear_sum_assignment(
        cost.detach().cpu().numpy()
    )
    return prediction_indices, target_indices, ious


@torch.no_grad()
def calculate_pipeline_entity_metrics(
    predictions,
    targets,
    symbol_iou_threshold=0.5,
    text_iou_threshold=0.5,
):
    """按照与V2-C一致的严格完整实体定义评估工业流水线。"""
    true_positive = 0
    false_positive = 0
    false_negative = 0
    breakdown_counts = create_entity_breakdown_counts()
    for prediction, target in zip(predictions, targets):
        predicted_symbol_boxes = prediction["symbol_boxes"]
        target_symbol_boxes = cxcywh_to_xyxy(
            target["symbol_boxes"].to(predicted_symbol_boxes.device)
        ).clamp(0, 1)
        target_text_boxes = cxcywh_to_xyxy(
            target["text_boxes"].to(predicted_symbol_boxes.device)
        ).clamp(0, 1)
        prediction_indices, target_indices, symbol_ious = _match_by_iou(
            predicted_symbol_boxes,
            target_symbol_boxes,
            symbol_iou_threshold,
        )
        text_ious = (
            box_iou(prediction["text_boxes"], target_text_boxes)
            if len(prediction["text_boxes"]) and len(target_text_boxes)
            else torch.empty(
                (len(prediction["text_boxes"]), len(target_text_boxes)),
                device=predicted_symbol_boxes.device,
            )
        )
        target_identifiers = [str(value) for value in target["identifiers"]]
        target_labels = target["symbol_labels"].to(predicted_symbol_boxes.device)
        target_locked = target["locked"].to(predicted_symbol_boxes.device).bool()
        breakdown_counts["num_targets"] += len(target_symbol_boxes)
        breakdown_counts["num_characters"] += sum(
            len(identifier) for identifier in target_identifiers
        )
        image_true_positive = 0
        for prediction_index, target_index in zip(
            prediction_indices,
            target_indices,
        ):
            symbol_iou = symbol_ious[prediction_index, target_index].item()
            text_iou = text_ious[prediction_index, target_index].item()
            symbol_box_correct = symbol_iou >= symbol_iou_threshold
            text_box_correct = text_iou >= text_iou_threshold
            symbol_class_correct = bool(
                prediction["symbol_labels"][prediction_index]
                == target_labels[target_index]
            )
            locked_correct = bool(
                prediction["locked"][prediction_index]
                == target_locked[target_index]
            )
            predicted_identifier = prediction["identifiers"][prediction_index]
            target_identifier = target_identifiers[target_index]
            identifier_exact_correct = predicted_identifier == target_identifier
            breakdown_counts["symbol_iou_sum"] += symbol_iou
            breakdown_counts["text_iou_sum"] += text_iou
            breakdown_counts["symbol_box_correct"] += symbol_box_correct
            breakdown_counts["text_box_correct"] += text_box_correct
            breakdown_counts["symbol_class_correct"] += symbol_class_correct
            breakdown_counts["locked_correct"] += locked_correct
            breakdown_counts["identifier_exact_correct"] += identifier_exact_correct
            identifier_distance = _levenshtein_distance(
                predicted_identifier,
                target_identifier,
            )
            breakdown_counts["identifier_character_correct"] += max(
                len(target_identifier) - identifier_distance,
                0,
            )
            if (
                symbol_box_correct
                and text_box_correct
                and symbol_class_correct
                and locked_correct
                and identifier_exact_correct
            ):
                image_true_positive += 1
        true_positive += image_true_positive
        false_positive += len(predicted_symbol_boxes) - image_true_positive
        false_negative += len(target_symbol_boxes) - image_true_positive
    metrics = summarize_counts(true_positive, false_positive, false_negative)
    metrics["breakdown_counts"] = breakdown_counts
    return metrics

