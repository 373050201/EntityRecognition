"""在P&ID测试集上执行按文字数量和首个失败环节划分的误差诊断。"""
import sys
from pathlib import Path
import torch
import yaml
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torchvision.ops import box_iou



PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import PIDEntityDataset
from model import build_model
from utils import (
    ctc_greedy_decode,
    cxcywh_to_xyxy,
    entity_collate_fn,
)



# 该脚本仅用于阶段性误差诊断；常规模型评估仍运行evaluation/evaluate.py。
checkpoint_name = "best.pt"
evaluation_split = None
duplicate_iou_threshold = 0.5
progress_interval = 50


def create_group_counts():
    """创建每个真实文字数量分组使用的累计计数。"""
    return {
        text_count: {
            "total": 0,
            "symbol_box_correct": 0,
            "symbol_class_correct": 0,
            "text_count_correct": 0,
            "all_text_boxes_correct": 0,
            "all_texts_correct": 0,
            "exact_entity": 0,
        }
        for text_count in range(4)
    }


def create_failure_counts():
    """创建互斥的首个失败环节计数。"""
    return {
        "symbol_box": 0,
        "symbol_class": 0,
        "text_count": 0,
        "text_box": 0,
        "text_recognition": 0,
        "exact": 0,
    }


def create_presence_counts():
    """创建文字槽位有效位的二分类计数。"""
    return {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
    }


def create_analysis_counts():
    """创建完整诊断任务的累计状态。"""
    return {
        "groups": create_group_counts(),
        "failure_stages": create_failure_counts(),
        "count_confusion": [
            [0 for _ in range(5)]
            for _ in range(4)
        ],
        "count_relation": {
            "under": 0,
            "correct": 0,
            "over": 0,
            "unmatched": 0,
        },
        "presence": create_presence_counts(),
        "duplicate_eligible_entities": 0,
        "duplicate_entities": 0,
        "empty_tiles": 0,
        "empty_tiles_with_false_positive": 0,
        "empty_tile_false_positives": 0,
    }


def safe_rate(numerator, denominator):
    """在分母为零时返回零，避免诊断表产生NaN。"""
    return numerator / denominator if denominator > 0 else 0.0


def build_model_from_config(config, checkpoint_path, device):
    """按主评估脚本的参数构造并加载模型。"""
    model = build_model(
        pretrained_weights=PROJECT_ROOT / config["pretrained_weights"],
        num_classes=config["num_classes"],
        num_queries=config["num_queries"],
        num_text_slots=config["num_text_slots"],
        sequence_length=config["ocr_sequence_length"],
        text_reference_offset_scale=config["text_reference_offset_scale"],
        text_reference_jitter_scale=config["text_reference_jitter_scale"],
        text_first_layer_sampling_scale=config[
            "text_first_layer_sampling_scale"
        ],
        text_refine_sampling_scale=config["text_refine_sampling_scale"],
        region_interaction_mode=config["region_interaction_mode"],
        region_interaction_init_scale=config[
            "region_interaction_init_scale"
        ],
        text_slot_interaction_mode=config["text_slot_interaction_mode"],
        text_slot_interaction_num_heads=config[
            "text_slot_interaction_num_heads"
        ],
        text_slot_interaction_init_scale=config[
            "text_slot_interaction_init_scale"
        ],
        ocr_guidance_mode=config["ocr_guidance_mode"],
        ocr_guidance_num_heads=config["ocr_guidance_num_heads"],
        ocr_guidance_gate_mode=config["ocr_guidance_gate_mode"],
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval(), checkpoint


def match_text_regions(
    predicted_text_boxes,
    predicted_texts,
    active_slots,
    target_text_boxes,
    target_texts,
    valid_text_indices,
    text_iou_threshold,
):
    """按主评估脚本的IoU Hungarian规则匹配实体内部文字集合。"""
    text_count_correct = len(active_slots) == len(valid_text_indices)
    all_text_boxes_correct = text_count_correct
    all_texts_correct = text_count_correct
    if len(active_slots) > 0 and len(valid_text_indices) > 0:
        source_boxes = cxcywh_to_xyxy(
            predicted_text_boxes[active_slots]
        ).clamp(0, 1)
        target_boxes = cxcywh_to_xyxy(
            target_text_boxes[valid_text_indices]
        ).clamp(0, 1)
        text_ious = box_iou(source_boxes, target_boxes)
        source_indices, local_target_indices = linear_sum_assignment(
            (1 - text_ious).cpu().numpy()
        )
        matched_ious = []
        for source_index, local_target_index in zip(
            source_indices,
            local_target_indices,
        ):
            matched_ious.append(
                text_ious[source_index, local_target_index].item()
            )
            slot_index = int(active_slots[source_index])
            text_index = int(valid_text_indices[local_target_index])
            all_texts_correct = (
                all_texts_correct
                and predicted_texts[slot_index] == target_texts[text_index]
            )
        all_text_boxes_correct = (
            all_text_boxes_correct
            and len(matched_ious) == len(valid_text_indices)
            and all(iou >= text_iou_threshold for iou in matched_ious)
        )
    elif len(valid_text_indices) > 0:
        all_text_boxes_correct = False
        all_texts_correct = False
    return text_count_correct, all_text_boxes_correct, all_texts_correct


def update_presence_counts(
    counts,
    predicted_presence,
    predicted_text_boxes,
    target_text_boxes,
    valid_text_indices,
):
    """按文字框IoU匹配生成槽位有效位标签并累计二分类计数。"""
    target_presence = torch.zeros_like(predicted_presence, dtype=torch.bool)
    if len(valid_text_indices) > 0:
        source_boxes = cxcywh_to_xyxy(predicted_text_boxes).clamp(0, 1)
        target_boxes = cxcywh_to_xyxy(
            target_text_boxes[valid_text_indices]
        ).clamp(0, 1)
        text_ious = box_iou(source_boxes, target_boxes)
        slot_indices, _ = linear_sum_assignment(
            (1 - text_ious).cpu().numpy()
        )
        target_presence[
            torch.as_tensor(slot_indices, device=target_presence.device)
        ] = True

    counts["true_positive"] += int(
        (predicted_presence & target_presence).sum().item()
    )
    counts["false_positive"] += int(
        (predicted_presence & ~target_presence).sum().item()
    )
    counts["false_negative"] += int(
        (~predicted_presence & target_presence).sum().item()
    )
    counts["true_negative"] += int(
        (~predicted_presence & ~target_presence).sum().item()
    )


def has_duplicate_active_boxes(
    predicted_text_boxes,
    active_slots,
    iou_threshold,
):
    """判断同一实体任意两个有效文字槽位是否产生高重叠框。"""
    if len(active_slots) < 2:
        return False
    active_boxes = cxcywh_to_xyxy(
        predicted_text_boxes[active_slots]
    ).clamp(0, 1)
    pairwise_ious = box_iou(active_boxes, active_boxes)
    upper_triangle = torch.triu(
        torch.ones_like(pairwise_ious, dtype=torch.bool),
        diagonal=1,
    )
    return bool((pairwise_ious[upper_triangle] >= iou_threshold).any())


def update_first_failure(counts, conditions):
    """按严格实体判定顺序累计互斥的首个失败环节。"""
    if not conditions["symbol_box_correct"]:
        counts["symbol_box"] += 1
    elif not conditions["symbol_class_correct"]:
        counts["symbol_class"] += 1
    elif not conditions["text_count_correct"]:
        counts["text_count"] += 1
    elif not conditions["all_text_boxes_correct"]:
        counts["text_box"] += 1
    elif not conditions["all_texts_correct"]:
        counts["text_recognition"] += 1
    else:
        counts["exact"] += 1


@torch.no_grad()
def analyze_batch(outputs, targets, config, counts):
    """使用与主评估一致的外层和内层匹配规则更新诊断统计。"""
    decoded_texts = ctc_greedy_decode(outputs["pred_text_logits"])
    batch_size, num_queries = outputs["pred_logits"].shape[:2]

    for batch_index in range(batch_size):
        foreground_scores, foreground_labels = outputs["pred_logits"][
            batch_index
        ].sigmoid().max(-1)
        keep = foreground_scores >= config["entity_conf_threshold"]
        kept_query_indices = torch.arange(
            num_queries,
            device=keep.device,
        )[keep]
        predicted_labels = foreground_labels[keep]
        predicted_presence = (
            outputs["pred_text_presence"][batch_index, keep].sigmoid()
            >= config["text_presence_threshold"]
        )
        predicted_symbol_boxes = cxcywh_to_xyxy(
            outputs["pred_symbol_boxes"][batch_index, keep]
        ).clamp(0, 1)
        predicted_text_boxes = outputs["pred_text_boxes"][batch_index, keep]
        predicted_texts = [
            decoded_texts[batch_index][query_index]
            for query_index in kept_query_indices.tolist()
        ]

        target_labels = targets[batch_index]["labels"].to(
            predicted_labels.device
        )
        target_symbol_boxes = cxcywh_to_xyxy(
            targets[batch_index]["symbol_boxes"].to(predicted_labels.device)
        ).clamp(0, 1)
        target_text_boxes = targets[batch_index]["text_boxes"].to(
            predicted_labels.device
        )
        target_text_valid = targets[batch_index]["text_valid"].to(
            predicted_labels.device
        )
        target_text_supervision = targets[batch_index][
            "text_supervision"
        ].to(predicted_labels.device)
        target_texts = targets[batch_index]["texts"]

        complete_indices = torch.where(target_text_supervision)[0].tolist()
        if len(complete_indices) == 0:
            counts["empty_tiles"] += 1

        prediction_indices = []
        target_indices = []
        valid_symbol_pairs = None
        symbol_ious = None
        if len(predicted_labels) > 0 and len(target_labels) > 0:
            symbol_ious = box_iou(
                predicted_symbol_boxes,
                target_symbol_boxes,
            )
            valid_symbol_pairs = (
                symbol_ious >= config["symbol_iou_threshold"]
            )
            matching_cost = torch.where(
                valid_symbol_pairs,
                1 - symbol_ious,
                1e3 + (1 - symbol_ious),
            )
            prediction_indices, target_indices = linear_sum_assignment(
                matching_cost.cpu().numpy()
            )
        target_to_prediction = {
            int(target_index): int(prediction_index)
            for prediction_index, target_index in zip(
                prediction_indices,
                target_indices,
            )
        }

        for target_index in complete_indices:
            valid_text_indices = torch.where(
                target_text_valid[target_index]
            )[0]
            true_text_count = len(valid_text_indices)
            group = counts["groups"][true_text_count]
            group["total"] += 1

            conditions = {
                "symbol_box_correct": False,
                "symbol_class_correct": False,
                "text_count_correct": False,
                "all_text_boxes_correct": False,
                "all_texts_correct": False,
            }
            prediction_index = target_to_prediction.get(target_index)
            if prediction_index is None:
                counts["count_confusion"][true_text_count][4] += 1
                counts["count_relation"]["unmatched"] += 1
                update_first_failure(counts["failure_stages"], conditions)
                continue

            conditions["symbol_box_correct"] = bool(
                valid_symbol_pairs[prediction_index, target_index]
            )
            conditions["symbol_class_correct"] = bool(
                predicted_labels[prediction_index]
                == target_labels[target_index]
            )
            active_slots = torch.where(
                predicted_presence[prediction_index]
            )[0]
            predicted_text_count = len(active_slots)
            counts["count_confusion"][true_text_count][
                predicted_text_count
            ] += 1
            if predicted_text_count < true_text_count:
                counts["count_relation"]["under"] += 1
            elif predicted_text_count > true_text_count:
                counts["count_relation"]["over"] += 1
            else:
                counts["count_relation"]["correct"] += 1

            (
                conditions["text_count_correct"],
                conditions["all_text_boxes_correct"],
                conditions["all_texts_correct"],
            ) = match_text_regions(
                predicted_text_boxes[prediction_index],
                predicted_texts[prediction_index],
                active_slots,
                target_text_boxes[target_index],
                target_texts[target_index],
                valid_text_indices,
                config["text_iou_threshold"],
            )
            if conditions["symbol_box_correct"]:
                update_presence_counts(
                    counts["presence"],
                    predicted_presence[prediction_index],
                    predicted_text_boxes[prediction_index],
                    target_text_boxes[target_index],
                    valid_text_indices,
                )

            if len(active_slots) >= 2:
                counts["duplicate_eligible_entities"] += 1
                counts["duplicate_entities"] += has_duplicate_active_boxes(
                    predicted_text_boxes[prediction_index],
                    active_slots,
                    duplicate_iou_threshold,
                )

            for condition_name in (
                "symbol_box_correct",
                "symbol_class_correct",
                "text_count_correct",
                "all_text_boxes_correct",
                "all_texts_correct",
            ):
                group[condition_name] += conditions[condition_name]
            exact_entity = all(conditions.values())
            group["exact_entity"] += exact_entity
            update_first_failure(counts["failure_stages"], conditions)

        if len(complete_indices) == 0:
            ignored_predictions = 0
            if valid_symbol_pairs is not None:
                ignored_predictions = sum(
                    not bool(target_text_supervision[target_index])
                    for prediction_index, target_index in zip(
                        prediction_indices,
                        target_indices,
                    )
                    if bool(valid_symbol_pairs[prediction_index, target_index])
                )
            false_positives = len(predicted_labels) - ignored_predictions
            counts["empty_tile_false_positives"] += false_positives
            counts["empty_tiles_with_false_positive"] += false_positives > 0


def aggregate_groups(groups, text_counts):
    """合并指定真实文字数量分组的计数。"""
    aggregate = {
        name: 0
        for name in next(iter(groups.values()))
    }
    for text_count in text_counts:
        for name, value in groups[text_count].items():
            aggregate[name] += value
    return aggregate


def print_group_results(groups):
    """输出按真实文字数量划分的条件正确率。"""
    print("\n按真实文字数量分组")
    print("GT文字数 | 实体数 | 图元框 | 图元类 | 数量 | 全部文框 | 全部文本 | GT完整正确")
    rows = [(str(index), groups[index]) for index in range(4)]
    rows.append(("2-3", aggregate_groups(groups, (2, 3))))
    rows.append(("全部", aggregate_groups(groups, (0, 1, 2, 3))))
    for label, row in rows:
        total = row["total"]
        print(
            f"{label:>8} | {total:>6} | "
            f"{safe_rate(row['symbol_box_correct'], total):.4f} | "
            f"{safe_rate(row['symbol_class_correct'], total):.4f} | "
            f"{safe_rate(row['text_count_correct'], total):.4f} | "
            f"{safe_rate(row['all_text_boxes_correct'], total):.4f} | "
            f"{safe_rate(row['all_texts_correct'], total):.4f} | "
            f"{safe_rate(row['exact_entity'], total):.4f}"
        )
    print(
        "说明：GT完整正确采用真实实体为分母，属于recall口径；"
        "不是主评估TP/(TP+FP+FN)的严格实体准确率。"
    )


def print_count_confusion(counts):
    """输出真实文字数量到预测有效槽位数的混淆矩阵。"""
    print("\n文字数量混淆矩阵（行归一化）")
    print("真实\\预测 | 0 | 1 | 2 | 3 | 未匹配实体")
    for true_count, row in enumerate(counts["count_confusion"]):
        total = sum(row)
        rates = [safe_rate(value, total) for value in row]
        print(
            f"{true_count:>9} | "
            + " | ".join(f"{rate:.4f}" for rate in rates)
        )

    relation = counts["count_relation"]
    relation_total = sum(relation.values())
    print("\n文字数量误差方向")
    print(
        f"少预测 {relation['under']} "
        f"({safe_rate(relation['under'], relation_total):.2%}) | "
        f"正确 {relation['correct']} "
        f"({safe_rate(relation['correct'], relation_total):.2%}) | "
        f"多预测 {relation['over']} "
        f"({safe_rate(relation['over'], relation_total):.2%}) | "
        f"实体未匹配 {relation['unmatched']} "
        f"({safe_rate(relation['unmatched'], relation_total):.2%})"
    )


def print_failure_results(counts):
    """输出互斥首个失败环节和附加诊断指标。"""
    failures = counts["failure_stages"]
    total = sum(failures.values())
    labels = {
        "symbol_box": "图元框/实体未匹配",
        "symbol_class": "图元分类",
        "text_count": "文字数量",
        "text_box": "文字框",
        "text_recognition": "文字识别",
        "exact": "GT完整正确",
    }
    print("\n首个失败环节（互斥归因）")
    for name, label in labels.items():
        print(
            f"{label:<16} {failures[name]:>6} "
            f"({safe_rate(failures[name], total):.2%})"
        )

    presence = counts["presence"]
    precision = safe_rate(
        presence["true_positive"],
        presence["true_positive"] + presence["false_positive"],
    )
    recall = safe_rate(
        presence["true_positive"],
        presence["true_positive"] + presence["false_negative"],
    )
    f1 = safe_rate(2 * precision * recall, precision + recall)
    print("\n槽位presence与几何分配的一致性（仅图元框匹配实体）")
    print(
        f"TP {presence['true_positive']} | "
        f"FP {presence['false_positive']} | "
        f"FN {presence['false_negative']} | "
        f"TN {presence['true_negative']}"
    )
    print(
        f"Precision {precision:.4f} | Recall {recall:.4f} | F1 {f1:.4f}"
    )
    print(
        "说明：该指标按三个槽位的文字框几何匹配生成槽位标签，"
        "用于诊断presence与框身份是否一致，不等同于实体数量准确率。"
    )

    print("\n重复框与空切片")
    print(
        "存在至少两个有效槽位的实体："
        f"{counts['duplicate_eligible_entities']}，"
        f"其中重复框实体：{counts['duplicate_entities']} "
        f"({safe_rate(counts['duplicate_entities'], counts['duplicate_eligible_entities']):.2%})"
    )
    print(
        f"无完整实体切片：{counts['empty_tiles']}，"
        f"出现误检的切片：{counts['empty_tiles_with_false_positive']} "
        f"({safe_rate(counts['empty_tiles_with_false_positive'], counts['empty_tiles']):.2%})，"
        f"误检实体总数：{counts['empty_tile_false_positives']}"
    )


config_path = PROJECT_ROOT / "config.yaml"
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

device = torch.device(
    "cuda" if str(config["device"]).lower() == "auto" and torch.cuda.is_available()
    else "cpu" if str(config["device"]).lower() == "auto"
    else config["device"]
)
split = evaluation_split or config["test_split"]
checkpoint_path = PROJECT_ROOT / checkpoint_name
if not checkpoint_path.is_file():
    raise FileNotFoundError(f"检查点不存在：{checkpoint_path}")

print(f"诊断设备：{device}")
print(f"诊断数据集：{split}")
print(f"检查点：{checkpoint_path}")
print(f"实体阈值：{config['entity_conf_threshold']}")
print(f"槽位阈值：{config['text_presence_threshold']}")

dataset = PIDEntityDataset(
    PROJECT_ROOT / config["dataset_root"],
    split=split,
    image_size=tuple(config["image_size"]),
    normalize=False,
    num_text_slots=config["num_text_slots"],
    include_partial_entities=config["include_partial_entities"],
)
data_loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
    collate_fn=entity_collate_fn,
)
model, checkpoint = build_model_from_config(
    config,
    checkpoint_path,
    device,
)
print(f"检查点轮次：{checkpoint.get('epoch', '未知')}")
print(f"测试切片：{len(dataset)}")

analysis_counts = create_analysis_counts()
with torch.inference_mode():
    for batch_index, (images, targets) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        outputs = model(images)
        analyze_batch(outputs, targets, config, analysis_counts)
        if batch_index % progress_interval == 0:
            print(f"诊断进度：{batch_index}/{len(data_loader)}")

print_group_results(analysis_counts["groups"])
print_count_confusion(analysis_counts)
print_failure_results(analysis_counts)
