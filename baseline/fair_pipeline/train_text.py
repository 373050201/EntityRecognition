"""训练公平流水线中独立的RT-DETRv2文字检测识别模型。"""
import time
from pathlib import Path
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import SCADAFairPipelineDataset
from losses import build_text_criterion
from models import build_text_spotter
from utils import (
    calculate_text_metrics,
    entity_collate_fn,
    seed_data_loader_worker,
    set_random_seed,
    summarize_counts,
)



# 设为"text_last.pt"可从本目录最近检查点续训。
resume_checkpoint_name = None


project_root = Path(__file__).resolve().parent
with open(project_root / "config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

set_random_seed(config["random_seed"])
if str(config["device"]).lower() == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(config["device"])
print(f"训练设备：{device}")

dataset_root = project_root / config["dataset_root"]
image_size = tuple(config["image_size"])
train_dataset = SCADAFairPipelineDataset(
    dataset_root,
    split=config["train_split"],
    image_size=image_size,
)
val_dataset = SCADAFairPipelineDataset(
    dataset_root,
    split=config["val_split"],
    image_size=image_size,
)
loader_generator = torch.Generator()
loader_generator.manual_seed(config["random_seed"])
train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
    collate_fn=entity_collate_fn,
    worker_init_fn=seed_data_loader_worker,
    generator=loader_generator,
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

model = build_text_spotter(
    pretrained_weights=project_root / config["pretrained_weights"],
    num_queries=config["num_queries"],
    sequence_length=config["ocr_sequence_length"],
).to(device)
criterion = build_text_criterion(
    bbox_coef=config["text_bbox_coef"],
    giou_coef=config["text_giou_coef"],
    ctc_coef=config["ctc_weight"],
).to(device)

backbone_parameters = list(model.backbone.parameters())
backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
other_parameters = [
    parameter
    for parameter in model.parameters()
    if id(parameter) not in backbone_parameter_ids
]
optimizer = AdamW(
    [
        {"params": other_parameters, "lr": config["learning_rate"]},
        {"params": backbone_parameters, "lr": config["backbone_learning_rate"]},
    ],
    weight_decay=config["weight_decay"],
)

best_model_path = project_root / config["text_best_checkpoint"]
last_model_path = project_root / config["text_last_checkpoint"]
start_epoch = 1
best_text_f1 = -1.0
best_val_loss = float("inf")
best_epoch = 0
early_stop_count = 0

if resume_checkpoint_name is not None:
    checkpoint = torch.load(
        project_root / resume_checkpoint_name,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer.param_groups[0]["lr"] = config["resume_learning_rate"]
    optimizer.param_groups[1]["lr"] = config["resume_backbone_learning_rate"]
    start_epoch = checkpoint["epoch"] + 1
    best_text_f1 = checkpoint["best_text_f1"]
    best_val_loss = checkpoint["best_val_loss"]
    best_epoch = checkpoint["best_epoch"]
    early_stop_count = checkpoint.get("early_stop_count", 0)
    print(f"从第{checkpoint['epoch']}轮恢复文字模型训练")

print("文字模型训练开始...")
start_time = time.time()
for epoch in range(start_epoch, config["epochs"] + 1):
    model.train()
    train_loss_sum = 0.0
    for images, targets in train_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        outputs = model(images, targets=targets, recognize_text=False)
        indices = criterion.matcher(outputs, targets)
        ocr_text_boxes = outputs["pred_boxes"]
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
            train_loss = train_loss - losses["loss_ctc"]
        optimizer.zero_grad()
        train_loss.backward()
        clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        optimizer.step()
        train_loss_sum += train_loss.detach().item()

    model.eval()
    val_loss_sum = 0.0
    text_metric_counts = {
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
            batch_text_metrics = calculate_text_metrics(
                outputs,
                targets,
                confidence_threshold=config["text_conf_threshold"],
                iou_threshold=config["text_iou_threshold"],
            )
            for name in text_metric_counts:
                text_metric_counts[name] += batch_text_metrics[name]

    train_loss = train_loss_sum / len(train_loader)
    val_loss = val_loss_sum / len(val_loader)
    text_metrics = summarize_counts(**text_metric_counts)
    print(
        f"第{epoch}轮 | 训练损失 {train_loss:.4f} | "
        f"验证损失 {val_loss:.4f} | "
        f"文字完整F1 {text_metrics['f1']:.4f}"
    )

    is_best = text_metrics["f1"] > best_text_f1 or (
        text_metrics["f1"] == best_text_f1 and val_loss < best_val_loss
    )
    if is_best:
        best_text_f1 = text_metrics["f1"]
        best_val_loss = val_loss
        best_epoch = epoch
        early_stop_count = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "text_metrics": text_metrics,
            },
            best_model_path,
        )
        print(f"已保存最佳文字模型：{best_model_path.name}")
    else:
        early_stop_count += 1

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "text_metrics": text_metrics,
            "best_text_f1": best_text_f1,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "early_stop_count": early_stop_count,
        },
        last_model_path,
    )
    if early_stop_count >= config["patience"]:
        print("文字模型早停触发")
        break

print(
    f"文字模型训练结束，用时{(time.time() - start_time) / 3600:.2f}h，"
    f"最佳轮次{best_epoch}，最佳文字完整F1 {best_text_f1:.4f}"
)
