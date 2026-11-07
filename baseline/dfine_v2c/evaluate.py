"""评估 D-FINE-V2C 检查点的实体指标、损失和推理效率。"""
import time
from pathlib import Path
import sys
import torch
import yaml
from thop import profile
from torch.utils.data import DataLoader
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import SCADAEntityDataset
from baseline.dfine_v2c.loss import build_criterion
from baseline.dfine_v2c.model import build_model
from utils import (
    calculate_entity_metrics,
    create_entity_breakdown_counts,
    entity_collate_fn,
    summarize_entity_breakdown,
    summarize_entity_metrics,
)



# 在这里修改评估选项
checkpoint_name = "best.pt"
evaluation_split = "synthetic_test"
use_gt_text_boxes = False
efficiency_batch_size = 1
efficiency_warmup_iterations = 10


def print_results(
    average_loss,
    component_losses,
    entity_metrics,
    entity_breakdown,
    efficiency_metrics,
):
    """以紧凑表格输出精度、损失和推理效率指标。"""
    print("实体指标   | F1 | Precision | Recall | 严格准确率")
    print(
        "           | "
        f"{entity_metrics['entity_f1']:.4f} | "
        f"{entity_metrics['entity_precision']:.4f} | "
        f"{entity_metrics['entity_recall']:.4f} | "
        f"{entity_metrics['entity_accuracy']:.4f}"
    )
    print(
        "实体计数   | "
        f"TP {entity_metrics['true_positive']} | "
        f"FP {entity_metrics['false_positive']} | "
        f"FN {entity_metrics['false_negative']}"
    )
    print("拆解指标   | 图IoU | 图框 | 文IoU | 文框 | 图类 | 锁 | 字符 | 编号")
    print(
        "           | "
        f"{entity_breakdown['symbol_mean_iou']:.4f} | "
        f"{entity_breakdown['symbol_box_accuracy']:.4f} | "
        f"{entity_breakdown['text_mean_iou']:.4f} | "
        f"{entity_breakdown['text_box_accuracy']:.4f} | "
        f"{entity_breakdown['symbol_class_accuracy']:.4f} | "
        f"{entity_breakdown['locked_accuracy']:.4f} | "
        f"{entity_breakdown['identifier_character_accuracy']:.4f} | "
        f"{entity_breakdown['identifier_exact_accuracy']:.4f}"
    )
    print("效率指标   | FPS | GFLOPs/图 | 峰值显存(GiB)")
    print(
        "           | "
        f"{efficiency_metrics['fps']:.2f} | "
        f"{efficiency_metrics['gflops_per_image']:.2f} | "
        f"{efficiency_metrics['peak_memory_gib']:.3f}"
    )
    print("评估集损失 | 图元框 | 文字框 | 图元分类 | 锁分类 | CTC | 总损失")
    print(
        "           | "
        f"{component_losses['symbol_box']:.4f} | "
        f"{component_losses['text_box']:.4f} | "
        f"{component_losses['class']:.4f} | "
        f"{component_losses['locked']:.4f} | "
        f"{component_losses['ctc']:.4f} | "
        f"{average_loss:.4f}"
    )


def measure_inference_efficiency(
    model,
    data_loader,
    device,
    warmup_iterations,
):
    """按单纯模型前向口径统计FPS、FLOPs和峰值显存。"""
    first_images, _ = next(iter(data_loader))
    first_images = first_images.to(
        device,
        non_blocking=device.type == "cuda",
    )

    macs, _ = profile(
        model,
        inputs=(first_images[:1],),
        verbose=False,
    )
    gflops_per_image = 2 * macs / 1e9

    print(f"预热模型前向：{warmup_iterations}次")
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(first_images)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    num_images = 0
    inference_seconds = 0.0
    with torch.inference_mode():
        for images, _ in data_loader:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_start_time = time.perf_counter()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - batch_start_time
            num_images += len(images)
            del outputs

    peak_memory_gib = (
        torch.cuda.max_memory_allocated(device) / 1024 ** 3
        if device.type == "cuda"
        else 0.0
    )
    return {
        "fps": num_images / inference_seconds,
        "gflops_per_image": gflops_per_image,
        "peak_memory_gib": peak_memory_gib,
    }


project_root = Path(__file__).resolve().parent
config_path = project_root / "config.yaml"

print("正在加载评估配置...")
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if str(config["device"]).lower() == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(config["device"])
print(f"评估设备：{device}")

split = evaluation_split or config["val_split"]
checkpoint_path = project_root / checkpoint_name
if not checkpoint_path.is_file():
    raise FileNotFoundError(f"检查点不存在：{checkpoint_path}")

print(f"正在加载数据集：{split}")
dataset = SCADAEntityDataset(
    project_root / config["dataset_root"],
    split=split,
    image_size=tuple(config["image_size"]),
    normalize=False,
)
data_loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
    collate_fn=entity_collate_fn,
)
efficiency_data_loader = DataLoader(
    dataset,
    batch_size=efficiency_batch_size,
    shuffle=False,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
    collate_fn=entity_collate_fn,
)
print(f"评估图片：{len(dataset)}")

print(f"正在加载检查点：{checkpoint_path}")
model = build_model(
    pretrained_weights=project_root / config["pretrained_weights"],
    num_classes=config["num_classes"],
    num_queries=config["num_queries"],
    sequence_length=config["ocr_sequence_length"],
    text_reference_offset_scale=config["text_reference_offset_scale"],
    text_reference_jitter_scale=config["text_reference_jitter_scale"],
    text_first_layer_sampling_scale=config[
        "text_first_layer_sampling_scale"
    ],
    text_refine_sampling_scale=config["text_refine_sampling_scale"],
    ocr_guidance_num_heads=config["ocr_guidance_num_heads"],
).to(device)
checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False,
)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
criterion = build_criterion(
    num_classes=config["num_classes"],
    text_bbox_coef=config["text_bbox_coef"],
    text_giou_coef=config["text_giou_coef"],
    text_fgl_coef=config["text_fgl_coef"],
    text_ddf_coef=config["text_ddf_coef"],
    locked_coef=config["locked_coef"],
    ctc_coef=config["ctc_weight"],
).to(device)
print(f"检查点轮次：{checkpoint.get('epoch', '未知')}")
print("模型范式：D-FINE-S + V2-C双区域双FDR")
print("OCR指导：固定Text-only")
ocr_box_mode = "真实文字框" if use_gt_text_boxes else "预测文字框"
print(f"OCR裁剪模式：{ocr_box_mode}")

model.eval()
print("开始统计推理效率...")
efficiency_metrics = measure_inference_efficiency(
    model,
    efficiency_data_loader,
    device,
    efficiency_warmup_iterations,
)
loss_sum = 0.0
component_loss_sums = {
    "symbol_box": 0.0,
    "text_box": 0.0,
    "class": 0.0,
    "locked": 0.0,
    "ctc": 0.0,
}
entity_counts = {
    "true_positive": 0,
    "false_positive": 0,
    "false_negative": 0,
}
entity_breakdown_counts = create_entity_breakdown_counts()

print("开始评估...")
start_time = time.time()
with torch.no_grad():
    for images, targets in data_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        outputs = model(images)

        if use_gt_text_boxes:
            # 只替换OCR的RoI输入，保留模型预测文字框用于损失和IoU评估。
            indices = criterion.matcher(outputs, targets)
            ocr_text_boxes = outputs["pred_text_boxes"].clone()
            for batch_index, (query_indices, target_indices) in enumerate(
                indices
            ):
                ocr_text_boxes[batch_index, query_indices] = targets[
                    batch_index
                ]["text_boxes"].to(device)[target_indices]
            outputs["pred_identifier_logits"] = model.text_recognition_head(
                outputs["ocr_features"],
                outputs["query_features"],
                outputs["text_query_features"],
                ocr_text_boxes,
            )

        losses = criterion(outputs, targets)

        loss_sum += losses["loss_total"].item()
        component_loss_sums["symbol_box"] += (
            losses["loss_symbol_bbox"]
            + losses["loss_symbol_giou"]
        ).item()
        component_loss_sums["text_box"] += (
            losses["loss_text_bbox"]
            + losses["loss_text_giou"]
            + losses["loss_text_aux"]
        ).item()
        component_loss_sums["class"] += losses["loss_class"].item()
        component_loss_sums["locked"] += losses["loss_locked"].item()
        component_loss_sums["ctc"] += losses[
            "loss_identifier_ctc"
        ].item()

        batch_metrics = calculate_entity_metrics(
            outputs,
            targets,
            entity_conf_threshold=config["entity_conf_threshold"],
            symbol_iou_threshold=config["symbol_iou_threshold"],
            text_iou_threshold=config["text_iou_threshold"],
            lock_threshold=config["lock_threshold"],
        )
        for count_name in entity_counts:
            entity_counts[count_name] += batch_metrics[count_name]
        for count_name in entity_breakdown_counts:
            entity_breakdown_counts[count_name] += batch_metrics[
                "breakdown_counts"
            ][count_name]

average_loss = loss_sum / len(data_loader)
component_losses = {
    name: value / len(data_loader)
    for name, value in component_loss_sums.items()
}
entity_metrics = summarize_entity_metrics(**entity_counts)
entity_breakdown = summarize_entity_breakdown(entity_breakdown_counts)
print_results(
    average_loss,
    component_losses,
    entity_metrics,
    entity_breakdown,
    efficiency_metrics,
)
print(f"评估用时：{time.time() - start_time:.2f}s")
