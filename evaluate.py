"""加载训练检查点，在指定数据划分上执行一次完整实体评估。"""
import time
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader
from dataset import SCADAEntityDataset
from loss import build_criterion
from model import build_model
from utils import (
    calculate_entity_metrics,
    create_entity_breakdown_counts,
    entity_collate_fn,
    summarize_entity_breakdown,
    summarize_entity_metrics,
)



# 在这里修改评估选项
checkpoint_name = "best.pt"
evaluation_split = "synthetic_val"
use_gt_text_boxes = False


def print_results(
    average_loss,
    component_losses,
    entity_metrics,
    entity_breakdown,
):
    """以紧凑表格输出损失、严格实体指标和各子任务通过率。"""
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
    print("实体指标   | 准确率 | 精确率 | 召回率 | F1")
    print(
        "           | "
        f"{entity_metrics['entity_accuracy']:.4f} | "
        f"{entity_metrics['entity_precision']:.4f} | "
        f"{entity_metrics['entity_recall']:.4f} | "
        f"{entity_metrics['entity_f1']:.4f}"
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
    region_interaction_mode=config["region_interaction_mode"],
    region_interaction_init_scale=config["region_interaction_init_scale"],
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
criterion = build_criterion(
    num_classes=config["num_classes"],
    text_bbox_coef=config["text_bbox_coef"],
    text_giou_coef=config["text_giou_coef"],
    locked_coef=config["locked_coef"],
    ctc_coef=config["ctc_weight"],
).to(device)
print(f"检查点轮次：{checkpoint.get('epoch', '未知')}")
print(f"区域交互模式：{config['region_interaction_mode']}")
print(f"OCR指导模式：{config['ocr_guidance_mode']}")
print(f"OCR指导门控：{config['ocr_guidance_gate_mode']}")
ocr_box_mode = "真实文字框" if use_gt_text_boxes else "预测文字框"
print(f"OCR裁剪模式：{ocr_box_mode}")

model.eval()
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
)
print(f"评估用时：{time.time() - start_time:.2f}s")
