"""
Entity-DETR模型训练。

联合训练图元分类、图元框、文字框、锁状态和编号CTC识别。
"""
import time
from pathlib import Path
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import SCADAEntityDataset
from .loss import build_criterion
from .model import build_model
from .utils import (
    calculate_entity_metrics,
    entity_collate_fn,
    summarize_entity_metrics,
)


baseline_root = Path(__file__).resolve().parent
project_root = baseline_root.parents[1]
config_path = baseline_root / "config.yaml"

print("正在加载训练配置...")
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

if str(config["device"]).lower() == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(config["device"])
print(f"训练设备：{device}")


print("正在加载训练/验证集...")
dataset_root = project_root / config["dataset_root"]
image_size = tuple(config["image_size"])
train_dataset = SCADAEntityDataset(
    dataset_root,
    split=config["train_split"],
    image_size=image_size,
)
val_dataset = SCADAEntityDataset(
    dataset_root,
    split=config["val_split"],
    image_size=image_size,
)
train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
    collate_fn=entity_collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
    collate_fn=entity_collate_fn,
)
print(f"训练图片：{len(train_dataset)}，验证图片：{len(val_dataset)}")


print("正在初始化模型...")
model = build_model(
    pretrained_backbone=config["pretrained_backbone"]
).to(device)
criterion = build_criterion(eos_coef=config["eos_coef"]).to(device)
criterion.weight_dict["loss_identifier_ctc"] = config["ctc_weight"]

# 预训练backbone使用更小学习率，其他模块使用主学习率。
backbone_parameters = list(model.backbone.parameters())
backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
other_parameters = [
    parameter
    for parameter in model.parameters()
    if id(parameter) not in backbone_parameter_ids
]
optimizer = AdamW(
    [
        {
            "params": other_parameters,
            "lr": config["learning_rate"],
        },
        {
            "params": backbone_parameters,
            "lr": config["backbone_learning_rate"],
        },
    ],
    weight_decay=config["weight_decay"],
)

best_model_path = baseline_root / "best.pt"
last_model_path = baseline_root / "last.pt"


print("训练开始...")
start_time = time.time()
best_entity_accuracy = -1.0
best_val_loss = float("inf")
best_epoch = 0
early_stop_count = 0

for epoch in range(1, config["epochs"] + 1):
    print(f"第{epoch}轮训练开始...")

    # 训练阶段
    model.train()
    train_loss_sum = 0.0
    for images, targets in train_loader:
        images = images.to(device, non_blocking=device.type == "cuda")

        # 先预测实体并完成匹配，再决定OCR使用真实框还是预测框。
        outputs = model(images, recognize_text=False)
        indices = criterion.matcher(outputs, targets)
        ocr_text_boxes = outputs["pred_text_boxes"]
        if epoch <= config["ocr_gt_box_epochs"]:
            ocr_text_boxes = ocr_text_boxes.clone()
            for batch_index, (query_indices, target_indices) in enumerate(indices):
                ocr_text_boxes[batch_index, query_indices] = targets[
                    batch_index
                ]["text_boxes"].to(device)[target_indices]

        outputs["pred_identifier_logits"] = model.text_recognition_head(
            outputs["ocr_features"],
            outputs["query_features"],
            ocr_text_boxes,
        )
        losses = criterion(outputs, targets, indices=indices)
        train_loss = losses["loss_total"]
        if epoch <= config["ctc_warmup_epochs"]:
            # 初期文字框尚不稳定，暂时不让CTC损失参与反向传播。
            train_loss = train_loss - (
                config["ctc_weight"] * losses["loss_identifier_ctc"]
            )

        optimizer.zero_grad()
        train_loss.backward()
        clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        optimizer.step()

        train_loss_sum += train_loss.detach().item()

    average_train_loss = train_loss_sum / len(train_loader)
    print(f"训练集损失：{average_train_loss:.4f}")

    # 验证阶段始终计算包含CTC在内的完整联合损失。
    model.eval()
    val_loss_sum = 0.0
    val_component_loss_sums = {
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
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=device.type == "cuda")

            outputs = model(images)
            losses = criterion(outputs, targets)
            val_loss_sum += losses["loss_total"].item()
            # 各分支损失乘以训练时的相同权重，因此它们相加等于总损失。
            val_component_loss_sums["symbol_box"] += (
                losses["loss_symbol_bbox"]
                * criterion.weight_dict["loss_symbol_bbox"]
                + losses["loss_symbol_giou"]
                * criterion.weight_dict["loss_symbol_giou"]
            ).item()
            val_component_loss_sums["text_box"] += (
                losses["loss_text_bbox"]
                * criterion.weight_dict["loss_text_bbox"]
                + losses["loss_text_giou"]
                * criterion.weight_dict["loss_text_giou"]
            ).item()
            val_component_loss_sums["class"] += (
                losses["loss_class"]
                * criterion.weight_dict["loss_class"]
            ).item()
            val_component_loss_sums["locked"] += (
                losses["loss_locked"]
                * criterion.weight_dict["loss_locked"]
            ).item()
            val_component_loss_sums["ctc"] += (
                losses["loss_identifier_ctc"]
                * criterion.weight_dict["loss_identifier_ctc"]
            ).item()

            batch_entity_metrics = calculate_entity_metrics(
                outputs,
                targets,
                entity_conf_threshold=config["entity_conf_threshold"],
                symbol_iou_threshold=config["symbol_iou_threshold"],
                text_iou_threshold=config["text_iou_threshold"],
                lock_threshold=config["lock_threshold"],
            )
            for count_name in entity_counts:
                entity_counts[count_name] += batch_entity_metrics[count_name]

    average_val_loss = val_loss_sum / len(val_loader)
    average_val_component_losses = {
        name: loss_sum / len(val_loader)
        for name, loss_sum in val_component_loss_sums.items()
    }
    val_entity_metrics = summarize_entity_metrics(**entity_counts)
    print("验证集损失 | 图元框 | 文字框 | 图元分类 | 锁分类 | CTC | 总损失")
    print(
        "           | "
        f"{average_val_component_losses['symbol_box']:.4f} | "
        f"{average_val_component_losses['text_box']:.4f} | "
        f"{average_val_component_losses['class']:.4f} | "
        f"{average_val_component_losses['locked']:.4f} | "
        f"{average_val_component_losses['ctc']:.4f} | "
        f"{average_val_loss:.4f}"
    )
    print(f"验证集实体准确率：{val_entity_metrics['entity_accuracy']:.4f}")

    # 每轮覆盖保存最新状态，便于诊断最后一轮并从中断处恢复训练。
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": average_val_loss,
            "val_loss_components": average_val_component_losses,
            "entity_metrics": val_entity_metrics,
        },
        last_model_path,
    )

    # 首先选择实体准确率最高的模型；准确率相同时选择验证损失更低的模型。
    entity_accuracy = val_entity_metrics["entity_accuracy"]
    improved = (
        entity_accuracy > best_entity_accuracy
        or (
            entity_accuracy == best_entity_accuracy
            and average_val_loss < best_val_loss
        )
    )
    if improved:
        best_entity_accuracy = entity_accuracy
        best_val_loss = average_val_loss
        best_epoch = epoch
        early_stop_count = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "entity_metrics": val_entity_metrics,
            },
            best_model_path,
        )
        print(f"已保存新的最佳模型：{best_model_path}")
    else:
        # 真实文字框引导期间不累计早停轮数，从切换到预测文字框后再开始计数。
        if epoch > config["ocr_gt_box_epochs"]:
            early_stop_count += 1
            if early_stop_count >= config["patience"]:
                print("早停触发！")
                break

print("训练结束")
elapsed_time = time.time() - start_time
print(f"总用时：{elapsed_time / 3600:.2f}h")
print(
    f"最佳训练轮次：{best_epoch}，"
    f"最佳实体准确率：{best_entity_accuracy:.4f}，"
    f"对应验证集损失：{best_val_loss:.4f}"
)
