"""加载两个独立模型，在测试集上评估匈牙利匹配后的完整实体流水线。"""
import time
from pathlib import Path
import torch
import yaml
from thop import profile
from torch.utils.data import DataLoader
from dataset import SCADAFairPipelineDataset
from losses import build_symbol_criterion, build_text_criterion
from matcher import associate_predictions
from models import build_symbol_detector, build_text_spotter
from utils import (
    calculate_pipeline_entity_metrics,
    create_entity_breakdown_counts,
    entity_collate_fn,
    summarize_counts,
    summarize_entity_breakdown,
)



# 设为None时使用config.yaml中的test_split。
evaluation_split = None


def measure_pipeline_efficiency(
    symbol_model,
    text_model,
    data_loader,
    device,
    config,
):
    """统计完整双模型前向及匈牙利关联的FPS、FLOPs和峰值显存。"""
    first_images, _ = next(iter(data_loader))
    first_images = first_images.to(
        device,
        non_blocking=device.type == "cuda",
    )
    symbol_macs, _ = profile(
        symbol_model,
        inputs=(first_images[:1],),
        verbose=False,
    )
    text_macs, _ = profile(
        text_model,
        inputs=(first_images[:1],),
        verbose=False,
    )
    gflops_per_image = 2 * (symbol_macs + text_macs) / 1e9

    warmup_iterations = config["efficiency_warmup_iterations"]
    print(f"预热完整流水线：{warmup_iterations}次")
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            symbol_outputs = symbol_model(first_images)
            text_outputs = text_model(first_images)
            associate_predictions(
                symbol_outputs,
                text_outputs,
                symbol_conf_threshold=config["symbol_conf_threshold"],
                text_conf_threshold=config["text_conf_threshold"],
                lock_threshold=config["lock_threshold"],
                max_distance=config["association_max_distance"],
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    num_images = 0
    elapsed_seconds = 0.0
    with torch.inference_mode():
        for images, _ in data_loader:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            symbol_outputs = symbol_model(images)
            text_outputs = text_model(images)
            associate_predictions(
                symbol_outputs,
                text_outputs,
                symbol_conf_threshold=config["symbol_conf_threshold"],
                text_conf_threshold=config["text_conf_threshold"],
                lock_threshold=config["lock_threshold"],
                max_distance=config["association_max_distance"],
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_seconds += time.perf_counter() - start_time
            num_images += len(images)

    peak_memory_gib = (
        torch.cuda.max_memory_allocated(device) / 1024 ** 3
        if device.type == "cuda"
        else 0.0
    )
    return {
        "fps": num_images / elapsed_seconds,
        "gflops_per_image": gflops_per_image,
        "peak_memory_gib": peak_memory_gib,
    }


project_root = Path(__file__).resolve().parent
with open(project_root / "config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
split = evaluation_split or config["test_split"]

if str(config["device"]).lower() == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(config["device"])
print(f"评估设备：{device}")
print(f"评估数据集：{split}")

dataset = SCADAFairPipelineDataset(
    project_root / config["dataset_root"],
    split=split,
    image_size=tuple(config["image_size"]),
)
data_loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
    collate_fn=entity_collate_fn,
)
efficiency_loader = DataLoader(
    dataset,
    batch_size=config["efficiency_batch_size"],
    shuffle=False,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
    collate_fn=entity_collate_fn,
)
print(f"评估图片：{len(dataset)}")

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

symbol_criterion = build_symbol_criterion(
    num_classes=config["num_symbol_classes"],
    bbox_coef=config["symbol_bbox_coef"],
    giou_coef=config["symbol_giou_coef"],
    locked_coef=config["locked_coef"],
).to(device)
text_criterion = build_text_criterion(
    bbox_coef=config["text_bbox_coef"],
    giou_coef=config["text_giou_coef"],
    ctc_coef=config["ctc_weight"],
).to(device)

print("开始统计完整流水线效率...")
efficiency_metrics = measure_pipeline_efficiency(
    symbol_model,
    text_model,
    efficiency_loader,
    device,
    config,
)

entity_counts = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
breakdown_counts = create_entity_breakdown_counts()
symbol_loss_sums = {
    "loss_class": 0.0,
    "loss_bbox": 0.0,
    "loss_giou": 0.0,
    "loss_locked": 0.0,
    "loss_aux": 0.0,
    "loss_total": 0.0,
}
text_loss_sums = {
    "loss_class": 0.0,
    "loss_bbox": 0.0,
    "loss_giou": 0.0,
    "loss_ctc": 0.0,
    "loss_aux": 0.0,
    "loss_total": 0.0,
}

print("开始评估完整实体...")
with torch.inference_mode():
    for images, targets in data_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        symbol_outputs = symbol_model(images)
        text_outputs = text_model(images)
        predictions = associate_predictions(
            symbol_outputs,
            text_outputs,
            symbol_conf_threshold=config["symbol_conf_threshold"],
            text_conf_threshold=config["text_conf_threshold"],
            lock_threshold=config["lock_threshold"],
            max_distance=config["association_max_distance"],
        )
        batch_metrics = calculate_pipeline_entity_metrics(
            predictions,
            targets,
            symbol_iou_threshold=config["symbol_iou_threshold"],
            text_iou_threshold=config["text_iou_threshold"],
        )
        for name in entity_counts:
            entity_counts[name] += batch_metrics[name]
        for name in breakdown_counts:
            breakdown_counts[name] += batch_metrics["breakdown_counts"][name]

        symbol_losses = symbol_criterion(symbol_outputs, targets)
        text_losses = text_criterion(text_outputs, targets)
        for name in symbol_loss_sums:
            symbol_loss_sums[name] += symbol_losses[name].item()
        for name in text_loss_sums:
            text_loss_sums[name] += text_losses[name].item()

entity_metrics = summarize_counts(**entity_counts)
entity_breakdown = summarize_entity_breakdown(breakdown_counts)
symbol_losses = {
    name: value / len(data_loader)
    for name, value in symbol_loss_sums.items()
}
text_losses = {
    name: value / len(data_loader)
    for name, value in text_loss_sums.items()
}

print("实体指标   | F1 | Precision | Recall | 严格准确率")
print(
    "           | "
    f"{entity_metrics['f1']:.4f} | "
    f"{entity_metrics['precision']:.4f} | "
    f"{entity_metrics['recall']:.4f} | "
    f"{entity_metrics['accuracy']:.4f}"
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
print("图元损失   | 分类 | L1框 | GIoU | 锁 | 辅助 | 总损失")
print(
    "           | "
    f"{symbol_losses['loss_class']:.4f} | "
    f"{symbol_losses['loss_bbox']:.4f} | "
    f"{symbol_losses['loss_giou']:.4f} | "
    f"{symbol_losses['loss_locked']:.4f} | "
    f"{symbol_losses['loss_aux']:.4f} | "
    f"{symbol_losses['loss_total']:.4f}"
)
print("文字损失   | 分类 | L1框 | GIoU | CTC | 辅助 | 总损失")
print(
    "           | "
    f"{text_losses['loss_class']:.4f} | "
    f"{text_losses['loss_bbox']:.4f} | "
    f"{text_losses['loss_giou']:.4f} | "
    f"{text_losses['loss_ctc']:.4f} | "
    f"{text_losses['loss_aux']:.4f} | "
    f"{text_losses['loss_total']:.4f}"
)
