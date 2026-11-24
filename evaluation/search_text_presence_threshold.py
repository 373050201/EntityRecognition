"""在验证集上缓存模型预测并搜索最佳文字槽位存在阈值。"""
import hashlib
import json
import sys
import time
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader



PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import PIDEntityDataset
from model import build_model
from utils import (
    BackgroundPrefetchLoader,
    calculate_entity_metrics,
    create_entity_breakdown_counts,
    ctc_greedy_decode,
    entity_collate_fn,
    summarize_entity_breakdown,
    summarize_entity_metrics,
)



# 在这里修改阈值搜索范围；先粗搜，再围绕粗搜最优值细搜。
checkpoint_name = "best.pt"
experiment_name = "pid_experiment_1"
coarse_step = 0.05
fine_radius = 0.05
fine_step = 0.001


def build_thresholds(start, end, step):
    """生成包含两端且消除浮点误差的阈值列表。"""
    num_steps = round((end - start) / step)
    return [round(start + index * step, 6) for index in range(num_steps + 1)]


def evaluate_threshold(cached_batches, config, threshold):
    """复用缓存预测，累计单个槽位阈值下的完整实体指标。"""
    entity_counts = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    breakdown_counts = create_entity_breakdown_counts()

    for cached_batch in cached_batches:
        batch_metrics = calculate_entity_metrics(
            cached_batch["outputs"],
            cached_batch["targets"],
            entity_conf_threshold=config["entity_conf_threshold"],
            symbol_iou_threshold=config["symbol_iou_threshold"],
            text_iou_threshold=config["text_iou_threshold"],
            text_presence_threshold=threshold,
            decoded_texts=cached_batch["decoded_texts"],
        )
        for count_name in entity_counts:
            entity_counts[count_name] += batch_metrics[count_name]
        for count_name in breakdown_counts:
            breakdown_counts[count_name] += batch_metrics[
                "breakdown_counts"
            ][count_name]

    result = {
        "text_presence_threshold": threshold,
        **summarize_entity_metrics(**entity_counts),
        **summarize_entity_breakdown(breakdown_counts),
    }
    return result


def select_best_result(results, reference_threshold):
    """优先最大化严格实体准确率，再依次比较F1和文字数量准确率。"""
    return max(
        results,
        key=lambda result: (
            result["entity_accuracy"],
            result["entity_f1"],
            result["text_count_accuracy"],
            -abs(result["text_presence_threshold"] - reference_threshold),
        ),
    )


project_root = PROJECT_ROOT
config_path = project_root / "config.yaml"
checkpoint_path = project_root / checkpoint_name
output_directory = project_root / "outputs" / experiment_name
output_path = output_directory / "threshold_search.json"

print("正在加载阈值搜索配置...")
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if str(config["device"]).lower() == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(config["device"])
if not checkpoint_path.is_file():
    raise FileNotFoundError(f"检查点不存在：{checkpoint_path}")

print(f"搜索设备：{device}")
print(f"验证集：{config['val_split']}")
dataset = PIDEntityDataset(
    project_root / config["dataset_root"],
    split=config["val_split"],
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
data_loader = BackgroundPrefetchLoader(
    data_loader,
    prefetch_batches=config.get("prefetch_batches", 2),
)
print(f"验证图片：{len(dataset)}")

print(f"正在加载检查点：{checkpoint_path}")
model = build_model(
    pretrained_weights=project_root / config["pretrained_weights"],
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
    region_interaction_init_scale=config["region_interaction_init_scale"],
    text_slot_interaction_mode=config["text_slot_interaction_mode"],
    text_slot_interaction_num_heads=config["text_slot_interaction_num_heads"],
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
model.eval()
print(f"检查点轮次：{checkpoint.get('epoch', '未知')}")

print("正在执行一次验证集推理并缓存必要预测...")
inference_start_time = time.time()
cached_batches = []
amp_enabled = bool(config.get("amp_enabled", False)) and device.type == "cuda"
with torch.inference_mode():
    for batch_index, (images, targets) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            outputs = model(images)

        cached_outputs = {
            "pred_logits": outputs["pred_logits"].detach().cpu(),
            "pred_symbol_boxes": outputs["pred_symbol_boxes"].detach().cpu(),
            "pred_text_boxes": outputs["pred_text_boxes"].detach().cpu(),
            "pred_text_presence": outputs["pred_text_presence"].detach().cpu(),
        }
        cached_batches.append(
            {
                "outputs": cached_outputs,
                "targets": targets,
                "decoded_texts": ctc_greedy_decode(
                    outputs["pred_text_logits"]
                ),
            }
        )
        if batch_index % 50 == 0 or batch_index == len(data_loader):
            print(f"推理进度：{batch_index}/{len(data_loader)}")
print(f"验证集推理用时：{time.time() - inference_start_time:.2f}s")

reference_threshold = float(config["text_presence_threshold"])
evaluated_results = {}
coarse_thresholds = build_thresholds(0.0, 1.0, coarse_step)
print(f"开始粗搜：{len(coarse_thresholds)}个阈值")
for threshold in coarse_thresholds:
    result = evaluate_threshold(cached_batches, config, threshold)
    evaluated_results[threshold] = result
    print(
        f"阈值{threshold:.3f} | "
        f"实体准确率{result['entity_accuracy']:.4f} | "
        f"F1 {result['entity_f1']:.4f} | "
        f"文字数量准确率{result['text_count_accuracy']:.4f}"
    )

coarse_best = select_best_result(
    list(evaluated_results.values()),
    reference_threshold,
)
fine_start = max(0.0, coarse_best["text_presence_threshold"] - fine_radius)
fine_end = min(1.0, coarse_best["text_presence_threshold"] + fine_radius)
fine_thresholds = build_thresholds(fine_start, fine_end, fine_step)
print(
    f"开始细搜：{fine_start:.3f}至{fine_end:.3f}，"
    f"共{len(fine_thresholds)}个阈值"
)
for threshold in fine_thresholds:
    if threshold in evaluated_results:
        continue
    result = evaluate_threshold(cached_batches, config, threshold)
    evaluated_results[threshold] = result

sorted_results = [
    evaluated_results[threshold]
    for threshold in sorted(evaluated_results)
]
best_result = select_best_result(sorted_results, reference_threshold)
checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
output = {
    "experiment_name": experiment_name,
    "checkpoint": checkpoint_name,
    "checkpoint_sha256": checkpoint_hash,
    "checkpoint_epoch": checkpoint.get("epoch"),
    "validation_split": config["val_split"],
    "num_validation_images": len(dataset),
    "fixed_thresholds": {
        "entity_conf_threshold": config["entity_conf_threshold"],
        "symbol_iou_threshold": config["symbol_iou_threshold"],
        "text_iou_threshold": config["text_iou_threshold"],
    },
    "reference_text_presence_threshold": reference_threshold,
    "selection_rule": (
        "maximize entity_accuracy, then entity_f1, then "
        "text_count_accuracy"
    ),
    "best_result": best_result,
    "results": sorted_results,
}
output_directory.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(output, file, ensure_ascii=False, indent=2)

print("阈值搜索结束")
print(f"最佳槽位阈值：{best_result['text_presence_threshold']:.3f}")
print(f"最佳实体准确率：{best_result['entity_accuracy']:.4f}")
print(f"对应F1：{best_result['entity_f1']:.4f}")
print(f"文字数量准确率：{best_result['text_count_accuracy']:.4f}")
print(f"完整结果：{output_path}")
