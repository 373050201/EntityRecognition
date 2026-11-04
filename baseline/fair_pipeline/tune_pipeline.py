"""在验证集上搜索公平流水线的置信度与匈牙利匹配距离门限。"""
import itertools
import re
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from dataset import SCADAFairPipelineDataset
from matcher import associate_predictions
from models import build_symbol_detector, build_text_spotter
from utils import (
    calculate_pipeline_entity_metrics,
    entity_collate_fn,
    summarize_counts,
)



# 先进行全局粗搜索，再围绕最优组合进行局部细搜索。
coarse_symbol_confidences = [0.30, 0.40, 0.50, 0.60, 0.70]
coarse_text_confidences = [0.30, 0.40, 0.50, 0.60, 0.70]
coarse_max_distances = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
fine_confidence_radius = 0.10
fine_confidence_step = 0.02
fine_distance_radius = 0.02
fine_distance_step = 0.005


def move_outputs_to_cpu(outputs, names):
    """仅缓存参数搜索所需的模型输出，避免占用额外显存。"""
    return {
        name: outputs[name].detach().cpu()
        for name in names
    }


def make_local_values(center, radius, step, lower, upper):
    """生成以粗搜索最优值为中心的去重细搜索序列。"""
    start = max(lower, center - radius)
    end = min(upper, center + radius)
    count = int(round((end - start) / step))
    return sorted({
        round(start + index * step, 6)
        for index in range(count + 1)
    } | {round(center, 6)})


def evaluate_parameters(
    cached_batches,
    config,
    symbol_confidence,
    text_confidence,
    max_distance,
):
    """在缓存的验证集输出上计算一组参数对应的完整实体指标。"""
    entity_counts = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    for symbol_outputs, text_outputs, targets in cached_batches:
        predictions = associate_predictions(
            symbol_outputs,
            text_outputs,
            symbol_conf_threshold=symbol_confidence,
            text_conf_threshold=text_confidence,
            lock_threshold=config["lock_threshold"],
            max_distance=max_distance,
        )
        batch_metrics = calculate_pipeline_entity_metrics(
            predictions,
            targets,
            symbol_iou_threshold=config["symbol_iou_threshold"],
            text_iou_threshold=config["text_iou_threshold"],
        )
        for name in entity_counts:
            entity_counts[name] += batch_metrics[name]
    return summarize_counts(**entity_counts)


def search_parameters(
    cached_batches,
    config,
    symbol_confidences,
    text_confidences,
    max_distances,
    stage_name,
):
    """遍历候选组合，以实体F1为首要标准返回最佳参数。"""
    combinations = list(itertools.product(
        symbol_confidences,
        text_confidences,
        max_distances,
    ))
    best_result = None
    print(f"{stage_name}：共{len(combinations)}组参数")
    for index, (
        symbol_confidence,
        text_confidence,
        max_distance,
    ) in enumerate(combinations, start=1):
        metrics = evaluate_parameters(
            cached_batches,
            config,
            symbol_confidence,
            text_confidence,
            max_distance,
        )
        result = {
            "symbol_conf_threshold": symbol_confidence,
            "text_conf_threshold": text_confidence,
            "association_max_distance": max_distance,
            "metrics": metrics,
        }
        # F1相同时依次偏好Precision、Recall及更严格的距离门限。
        result_key = (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            metrics["accuracy"],
            -max_distance,
            symbol_confidence,
            text_confidence,
        )
        if best_result is None or result_key > best_result["key"]:
            result["key"] = result_key
            best_result = result
        if index % 50 == 0 or index == len(combinations):
            print(
                f"  已完成 {index}/{len(combinations)}，"
                f"当前最佳实体F1 {best_result['metrics']['f1']:.4f}"
            )
    return best_result


def update_config_file(config_path, best_result):
    """保留原配置格式，将验证集选出的三个门限写回配置文件。"""
    config_text = config_path.read_text(encoding="utf-8")
    for name in (
        "symbol_conf_threshold",
        "text_conf_threshold",
        "association_max_distance",
    ):
        value = best_result[name]
        config_text, replacement_count = re.subn(
            rf"(?m)^{name}:\s*[^#\r\n]+",
            f"{name}: {value:.6g}",
            config_text,
            count=1,
        )
        if replacement_count != 1:
            raise RuntimeError(f"config.yaml中未唯一找到参数：{name}")
    config_path.write_text(config_text, encoding="utf-8")


project_root = Path(__file__).resolve().parent
config_path = project_root / "config.yaml"
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if str(config["device"]).lower() == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(config["device"])
print(f"参数搜索设备：{device}")
print(f"参数搜索数据集：{config['val_split']}")

val_dataset = SCADAFairPipelineDataset(
    project_root / config["dataset_root"],
    split=config["val_split"],
    image_size=tuple(config["image_size"]),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
    collate_fn=entity_collate_fn,
)
print(f"验证图片：{len(val_dataset)}")

symbol_checkpoint_path = project_root / config["symbol_best_checkpoint"]
text_checkpoint_path = project_root / config["text_best_checkpoint"]
if not symbol_checkpoint_path.is_file():
    raise FileNotFoundError(f"图元检查点不存在：{symbol_checkpoint_path}")
if not text_checkpoint_path.is_file():
    raise FileNotFoundError(f"文字检查点不存在：{text_checkpoint_path}")

symbol_model = build_symbol_detector(
    num_classes=config["num_symbol_classes"],
    num_queries=config["num_queries"],
).to(device)
text_model = build_text_spotter(
    num_queries=config["num_queries"],
    sequence_length=config["ocr_sequence_length"],
).to(device)
symbol_checkpoint = torch.load(
    symbol_checkpoint_path,
    map_location=device,
    weights_only=False,
)
text_checkpoint = torch.load(
    text_checkpoint_path,
    map_location=device,
    weights_only=False,
)
symbol_model.load_state_dict(symbol_checkpoint["model_state_dict"], strict=True)
text_model.load_state_dict(text_checkpoint["model_state_dict"], strict=True)
symbol_model.eval()
text_model.eval()
print(f"图元模型轮次：{symbol_checkpoint.get('epoch', '未知')}")
print(f"文字模型轮次：{text_checkpoint.get('epoch', '未知')}")

# 模型只前向一次，后续搜索全部复用CPU缓存。
cached_batches = []
print("开始缓存验证集模型输出...")
with torch.inference_mode():
    for images, targets in val_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        symbol_outputs = move_outputs_to_cpu(
            symbol_model(images),
            ("pred_logits", "pred_boxes", "pred_locked"),
        )
        text_outputs = move_outputs_to_cpu(
            text_model(images),
            ("pred_logits", "pred_boxes", "pred_identifier_logits"),
        )
        cached_batches.append((symbol_outputs, text_outputs, targets))
del symbol_model, text_model
if device.type == "cuda":
    torch.cuda.empty_cache()
print(f"已缓存{len(cached_batches)}个批次")

coarse_best = search_parameters(
    cached_batches,
    config,
    coarse_symbol_confidences,
    coarse_text_confidences,
    coarse_max_distances,
    "粗搜索",
)
fine_symbol_confidences = make_local_values(
    coarse_best["symbol_conf_threshold"],
    fine_confidence_radius,
    fine_confidence_step,
    0.01,
    0.99,
)
fine_text_confidences = make_local_values(
    coarse_best["text_conf_threshold"],
    fine_confidence_radius,
    fine_confidence_step,
    0.01,
    0.99,
)
fine_max_distances = make_local_values(
    coarse_best["association_max_distance"],
    fine_distance_radius,
    fine_distance_step,
    0.0,
    1.0,
)
best_result = search_parameters(
    cached_batches,
    config,
    fine_symbol_confidences,
    fine_text_confidences,
    fine_max_distances,
    "细搜索",
)
update_config_file(config_path, best_result)

best_metrics = best_result["metrics"]
print("验证集参数搜索完成，最佳参数已写入config.yaml：")
print(
    f"symbol_conf_threshold: {best_result['symbol_conf_threshold']:.4f}\n"
    f"text_conf_threshold: {best_result['text_conf_threshold']:.4f}\n"
    f"association_max_distance: {best_result['association_max_distance']:.4f}"
)
print("验证实体指标 | F1 | Precision | Recall | 严格准确率")
print(
    "             | "
    f"{best_metrics['f1']:.4f} | "
    f"{best_metrics['precision']:.4f} | "
    f"{best_metrics['recall']:.4f} | "
    f"{best_metrics['accuracy']:.4f}"
)
print(
    "验证实体计数 | "
    f"TP {best_metrics['true_positive']} | "
    f"FP {best_metrics['false_positive']} | "
    f"FN {best_metrics['false_negative']}"
)
