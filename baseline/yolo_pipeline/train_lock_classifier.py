"""训练MobileNetV3-Small图元锁状态二分类器。"""
import time
from pathlib import Path
import torch
import yaml
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from lock_classifier import build_lock_classifier, build_lock_transform
from lock_dataset import LockStateDataset
from utils import set_random_seed, summarize_counts



yolo_dir = Path(__file__).resolve().parent
with open(yolo_dir / "config_yolo.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
set_random_seed(config["random_seed"])
device = torch.device(
    f"cuda:{config['device']}"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"锁状态分类训练设备：{device}")

dataset_root = yolo_dir / config["dataset_root"]
train_dataset = LockStateDataset(
    dataset_root,
    config["train_split"],
    build_lock_transform(config["lock_input_size"], training=True),
    config["lock_crop_padding_ratio"],
)
val_dataset = LockStateDataset(
    dataset_root,
    config["val_split"],
    build_lock_transform(config["lock_input_size"], training=False),
    config["lock_crop_padding_ratio"],
)
train_loader = DataLoader(
    train_dataset,
    batch_size=config["lock_batch_size"],
    shuffle=True,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["lock_batch_size"],
    shuffle=False,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
)
class_counts = train_dataset.class_counts()
class_weights = torch.tensor(
    [
        len(train_dataset) / (2 * count)
        for count in class_counts
    ],
    dtype=torch.float32,
    device=device,
)
print(
    f"锁状态训练样本：未锁{class_counts[0]}，已锁{class_counts[1]}；"
    f"验证样本：{len(val_dataset)}"
)

model = build_lock_classifier(pretrained=True).to(device)
classifier_parameters = list(model.classifier.parameters())
classifier_parameter_ids = {id(parameter) for parameter in classifier_parameters}
backbone_parameters = [
    parameter
    for parameter in model.parameters()
    if id(parameter) not in classifier_parameter_ids
]
optimizer = AdamW(
    [
        {
            "params": classifier_parameters,
            "lr": config["lock_learning_rate"],
        },
        {
            "params": backbone_parameters,
            "lr": config["lock_backbone_learning_rate"],
        },
    ],
    weight_decay=config["lock_weight_decay"],
)
criterion = nn.CrossEntropyLoss(weight=class_weights)
best_model_path = yolo_dir / config["lock_best_checkpoint"]
last_model_path = yolo_dir / config["lock_last_checkpoint"]
best_f1 = -1.0
best_val_loss = float("inf")
best_epoch = 0
early_stop_count = 0

print("锁状态分类器训练开始...")
start_time = time.time()
for epoch in range(1, config["lock_epochs"] + 1):
    model.train()
    train_loss_sum = 0.0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss_sum += loss.detach().item()

    model.eval()
    val_loss_sum = 0.0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    with torch.inference_mode():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            logits = model(images)
            val_loss_sum += criterion(logits, labels).item()
            predictions = logits.argmax(dim=-1)
            true_positive += int(((predictions == 1) & (labels == 1)).sum())
            false_positive += int(((predictions == 1) & (labels == 0)).sum())
            false_negative += int(((predictions == 0) & (labels == 1)).sum())

    train_loss = train_loss_sum / len(train_loader)
    val_loss = val_loss_sum / len(val_loader)
    metrics = summarize_counts(true_positive, false_positive, false_negative)
    print(
        f"第{epoch}轮 | 训练损失 {train_loss:.4f} | "
        f"验证损失 {val_loss:.4f} | 锁状态F1 {metrics['f1']:.4f}"
    )
    is_best = metrics["f1"] > best_f1 or (
        metrics["f1"] == best_f1 and val_loss < best_val_loss
    )
    if is_best:
        best_f1 = metrics["f1"]
        best_val_loss = val_loss
        best_epoch = epoch
        early_stop_count = 0
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "lock_metrics": metrics,
            },
            best_model_path,
        )
        print(f"已保存最佳锁状态分类器：{best_model_path.name}")
    else:
        early_stop_count += 1

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "lock_metrics": metrics,
            "best_f1": best_f1,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "early_stop_count": early_stop_count,
        },
        last_model_path,
    )
    if early_stop_count >= config["lock_patience"]:
        print("锁状态分类器早停触发")
        break

print(
    f"锁状态分类器训练结束，用时{(time.time() - start_time) / 60:.2f}min，"
    f"最佳轮次{best_epoch}，最佳锁状态F1 {best_f1:.4f}"
)

