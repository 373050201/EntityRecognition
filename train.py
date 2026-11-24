"""基于官方 RT-DETRv2-R18 的实体模型训练。"""
import time
from pathlib import Path
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from dataset import PIDEntityDataset
from loss import build_criterion
from model import build_model
from utils import (
    BackgroundPrefetchLoader,
    calculate_entity_metrics,
    capture_random_state,
    create_entity_breakdown_counts,
    entity_collate_fn,
    restore_random_state,
    seed_data_loader_worker,
    set_random_seed,
    summarize_entity_breakdown,
    summarize_entity_metrics,
)



# 训练模式开关：设为None从头训练；设为"last.pt"从断点继续训练。
resume_checkpoint_name = None


project_root = Path(__file__).resolve().parent
config_path = project_root / "config.yaml"

print("正在加载训练配置...")
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

# 必须在模型初始化和DataLoader创建前固定种子。
set_random_seed(config["random_seed"])
print(f"训练随机种子：{config['random_seed']}")
print(f"区域交互模式：{config['region_interaction_mode']}")
print(f"文字槽位交互模式：{config['text_slot_interaction_mode']}")
print(f"OCR指导模式：{config['ocr_guidance_mode']}")
print(f"OCR指导门控：{config['ocr_guidance_gate_mode']}")

if str(config["device"]).lower() == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(config["device"])
print(f"训练设备：{device}")

amp_enabled = bool(config["amp_enabled"] and device.type == "cuda")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
print(f"混合精度：{amp_enabled}")


print("正在加载训练/验证集...")
dataset_root = project_root / config["dataset_root"]
image_size = tuple(config["image_size"])
train_dataset = PIDEntityDataset(
    dataset_root,
    split=config["train_split"],
    image_size=image_size,
    normalize=False,
    num_text_slots=config["num_text_slots"],
    include_partial_entities=config["include_partial_entities"],
)
val_dataset = PIDEntityDataset(
    dataset_root,
    split=config["val_split"],
    image_size=image_size,
    normalize=False,
    num_text_slots=config["num_text_slots"],
    include_partial_entities=config["include_partial_entities"],
)
train_loader_generator = torch.Generator()
train_loader_generator.manual_seed(config["random_seed"])
val_loader_generator = torch.Generator()
val_loader_generator.manual_seed(config["random_seed"] + 1)
train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
    collate_fn=entity_collate_fn,
    worker_init_fn=seed_data_loader_worker,
    generator=train_loader_generator,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["workers"],
    pin_memory=device.type == "cuda",
    collate_fn=entity_collate_fn,
    worker_init_fn=seed_data_loader_worker,
    generator=val_loader_generator,
)
train_loader = BackgroundPrefetchLoader(
    train_loader,
    prefetch_batches=config["prefetch_batches"],
)
val_loader = BackgroundPrefetchLoader(
    val_loader,
    prefetch_batches=config["prefetch_batches"],
)
print(f"训练图片：{len(train_dataset)}，验证图片：{len(val_dataset)}")


print("正在初始化模型...")
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
criterion = build_criterion(
    num_classes=config["num_classes"],
    text_bbox_coef=config["text_bbox_coef"],
    text_giou_coef=config["text_giou_coef"],
    text_presence_coef=config["text_presence_coef"],
    text_duplicate_coef=config["text_duplicate_coef"],
    text_duplicate_iou_threshold=config["text_duplicate_iou_threshold"],
    ctc_coef=config["ctc_weight"],
).to(device)

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
    fused=device.type == "cuda",
)
grad_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

best_model_path = project_root / "best.pt"
last_model_path = project_root / "last.pt"


print("训练开始...")
start_time = time.time()
best_entity_accuracy = -1.0
best_val_loss = float("inf")
best_epoch = 0
early_stop_count = 0
start_epoch = 1

if resume_checkpoint_name is not None:
    resume_checkpoint_path = project_root / resume_checkpoint_name
    if not resume_checkpoint_path.is_file():
        raise FileNotFoundError(f"续训检查点不存在：{resume_checkpoint_path}")

    resume_checkpoint = torch.load(
        resume_checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "optimizer_state_dict" not in resume_checkpoint:
        raise KeyError("续训检查点不包含优化器状态，请使用last.pt而不是best.pt")

    model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
    if "grad_scaler_state_dict" in resume_checkpoint:
        grad_scaler.load_state_dict(
            resume_checkpoint["grad_scaler_state_dict"]
        )
    start_epoch = resume_checkpoint["epoch"] + 1

    # 新版last.pt会恢复随机序列；旧检查点仍可续训，但不能保证与原运行逐步一致。
    if "random_state" in resume_checkpoint:
        restore_random_state(
            resume_checkpoint["random_state"],
            train_loader_generator,
        )
        print("已恢复检查点中的随机数状态")
    else:
        print("当前检查点未保存随机数状态，将从配置种子继续")

    # 加载优化器会恢复旧学习率，续训时再显式切换为较小学习率。
    optimizer.param_groups[0]["lr"] = config["resume_learning_rate"]
    optimizer.param_groups[1]["lr"] = config[
        "resume_backbone_learning_rate"
    ]

    if "best_entity_accuracy" in resume_checkpoint:
        best_entity_accuracy = resume_checkpoint["best_entity_accuracy"]
        best_val_loss = resume_checkpoint["best_val_loss"]
        best_epoch = resume_checkpoint["best_epoch"]
        early_stop_count = resume_checkpoint.get("early_stop_count", 0)
    elif best_model_path.is_file():
        # 兼容旧版last.pt：从best.pt恢复历史最佳指标。
        best_checkpoint = torch.load(
            best_model_path,
            map_location="cpu",
            weights_only=False,
        )
        best_entity_accuracy = best_checkpoint["entity_metrics"][
            "entity_accuracy"
        ]
        best_val_loss = best_checkpoint["val_loss"]
        best_epoch = best_checkpoint["epoch"]
    else:
        best_entity_accuracy = resume_checkpoint["entity_metrics"][
            "entity_accuracy"
        ]
        best_val_loss = resume_checkpoint["val_loss"]
        best_epoch = resume_checkpoint["epoch"]

    print(
        f"已从第{resume_checkpoint['epoch']}轮恢复训练，"
        f"下一轮为第{start_epoch}轮"
    )
    print(
        f"历史最佳：第{best_epoch}轮，"
        f"实体准确率{best_entity_accuracy:.4f}，"
        f"验证损失{best_val_loss:.4f}"
    )
    print(
        "续训学习率："
        f"普通参数{optimizer.param_groups[0]['lr']:.2e}，"
        f"backbone参数{optimizer.param_groups[1]['lr']:.2e}"
    )
else:
    print("训练模式：从头训练")

if start_epoch > config["epochs"]:
    raise ValueError(
        f"断点下一轮为{start_epoch}，"
        f"但配置的总轮数只有{config['epochs']}，请增大epochs"
    )

for epoch in range(start_epoch, config["epochs"] + 1):
    print(f"第{epoch}轮训练开始...")

    # 训练阶段
    model.train()
    train_loss_sum = torch.zeros((), device=device)
    epoch_train_start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for batch_index, (images, targets) in enumerate(train_loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            # 先预测实体并完成匹配，再决定OCR使用真实框还是预测框。
            outputs = model(images, targets=targets, recognize_text=False)
            indices = criterion.matcher(outputs, targets)
            text_matches = criterion.match_text_slots(outputs, targets, indices)
            ocr_text_boxes = outputs["pred_text_boxes"]
            if epoch <= config["ocr_gt_box_epochs"]:
                ocr_text_boxes = ocr_text_boxes.clone()
                for target_batch_index, matches in enumerate(text_matches):
                    ocr_text_boxes[
                        target_batch_index,
                        matches["query_indices"],
                        matches["slot_indices"],
                    ] = targets[target_batch_index]["text_boxes"].to(device)[
                        matches["target_indices"],
                        matches["text_indices"],
                    ]

            # CTC只使用匹配文字，跳过没有梯度贡献的未匹配OCR区域。
            outputs["matched_text_logits"] = (
                model.text_recognition_head.forward_selected(
                    outputs["ocr_features"],
                    outputs["query_features"],
                    outputs["text_query_features"],
                    ocr_text_boxes,
                    text_matches,
                )
            )
            losses = criterion(
                outputs,
                targets,
                indices=indices,
                text_matches=text_matches,
            )
            train_loss = losses["loss_total"]
            if epoch <= config["ctc_warmup_epochs"]:
                # 初期文字框尚不稳定，暂时不让CTC损失参与反向传播。
                train_loss = train_loss - (
                    losses["loss_text_ctc"]
                )

        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(train_loss).backward()
        grad_scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        grad_scaler.step(optimizer)
        grad_scaler.update()

        # 保持损失在GPU上累计，避免每个batch调用item导致强制同步。
        train_loss_sum += train_loss.detach()
        if batch_index % config["log_interval"] == 0:
            elapsed_time = time.time() - epoch_train_start_time
            seconds_per_batch = elapsed_time / batch_index
            remaining_hours = (
                (len(train_loader) - batch_index)
                * seconds_per_batch
                / 3600
            )
            peak_memory_gib = (
                torch.cuda.max_memory_allocated() / 1024 ** 3
                if device.type == "cuda" else 0.0
            )
            print(
                f"训练进度：{batch_index}/{len(train_loader)}，"
                f"{seconds_per_batch:.3f}s/batch，"
                f"本轮剩余约{remaining_hours:.2f}h，"
                f"峰值显存{peak_memory_gib:.2f}GiB"
            )

    average_train_loss = (train_loss_sum / len(train_loader)).item()
    print(f"训练集损失：{average_train_loss:.4f}")

    # 验证阶段始终计算包含CTC在内的完整联合损失。
    model.eval()
    val_loss_sum = 0.0
    val_component_loss_sums = {
        "symbol_box": 0.0,
        "text_box": 0.0,
        "text_presence": 0.0,
        "text_duplicate": 0.0,
        "class": 0.0,
        "ctc": 0.0,
    }
    entity_counts = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    entity_breakdown_counts = create_entity_breakdown_counts()
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=device.type == "cuda")

            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
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
                + losses["loss_text_aux"]
                * criterion.weight_dict["loss_text_aux"]
            ).item()
            val_component_loss_sums["class"] += (
                losses["loss_class"]
                * criterion.weight_dict["loss_class"]
            ).item()
            val_component_loss_sums["text_presence"] += (
                losses["loss_text_presence"]
                * criterion.weight_dict["loss_text_presence"]
            ).item()
            val_component_loss_sums["text_duplicate"] += (
                losses["loss_text_duplicate"]
                * criterion.weight_dict["loss_text_duplicate"]
            ).item()
            val_component_loss_sums["ctc"] += (
                losses["loss_text_ctc"]
                * criterion.weight_dict["loss_text_ctc"]
            ).item()

            batch_entity_metrics = calculate_entity_metrics(
                outputs,
                targets,
                entity_conf_threshold=config["entity_conf_threshold"],
                symbol_iou_threshold=config["symbol_iou_threshold"],
                text_iou_threshold=config["text_iou_threshold"],
                text_presence_threshold=config["text_presence_threshold"],
            )
            for count_name in entity_counts:
                entity_counts[count_name] += batch_entity_metrics[count_name]
            for count_name in entity_breakdown_counts:
                entity_breakdown_counts[count_name] += batch_entity_metrics[
                    "breakdown_counts"
                ][count_name]

    average_val_loss = val_loss_sum / len(val_loader)
    average_val_component_losses = {
        name: loss_sum / len(val_loader)
        for name, loss_sum in val_component_loss_sums.items()
    }
    val_entity_metrics = summarize_entity_metrics(**entity_counts)
    val_entity_breakdown = summarize_entity_breakdown(entity_breakdown_counts)
    print("验证集损失 | 图元框 | 文字框 | 槽位存在 | 重复抑制 | 图元分类 | CTC | 总损失")
    print(
        "           | "
        f"{average_val_component_losses['symbol_box']:.4f} | "
        f"{average_val_component_losses['text_box']:.4f} | "
        f"{average_val_component_losses['text_presence']:.4f} | "
        f"{average_val_component_losses['text_duplicate']:.4f} | "
        f"{average_val_component_losses['class']:.4f} | "
        f"{average_val_component_losses['ctc']:.4f} | "
        f"{average_val_loss:.4f}"
    )
    print(f"验证集实体准确率：{val_entity_metrics['entity_accuracy']:.4f}")
    print("验证集拆解 | 图IoU | 图框 | 文IoU | 文框 | 图类 | 数量 | 字符 | 文本")
    print(
        "           | "
        f"{val_entity_breakdown['symbol_mean_iou']:.4f} | "
        f"{val_entity_breakdown['symbol_box_accuracy']:.4f} | "
        f"{val_entity_breakdown['text_mean_iou']:.4f} | "
        f"{val_entity_breakdown['text_box_accuracy']:.4f} | "
        f"{val_entity_breakdown['symbol_class_accuracy']:.4f} | "
        f"{val_entity_breakdown['text_count_accuracy']:.4f} | "
        f"{val_entity_breakdown['text_character_accuracy']:.4f} | "
        f"{val_entity_breakdown['text_exact_accuracy']:.4f}"
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
    should_stop = False
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
                "entity_breakdown": val_entity_breakdown,
                "random_seed": config["random_seed"],
            },
            best_model_path,
        )
        print(f"已保存新的最佳模型：{best_model_path}")
    else:
        # 真实文字框引导期间不累计早停轮数，从切换到预测文字框后再开始计数。
        if epoch > config["ocr_gt_box_epochs"]:
            early_stop_count += 1
            if early_stop_count >= config["patience"]:
                should_stop = True

    # 每轮覆盖保存最新状态和历史最佳状态，供下一次完整恢复训练。
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "grad_scaler_state_dict": grad_scaler.state_dict(),
            "val_loss": average_val_loss,
            "val_loss_components": average_val_component_losses,
            "entity_metrics": val_entity_metrics,
            "entity_breakdown": val_entity_breakdown,
            "best_entity_accuracy": best_entity_accuracy,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "early_stop_count": early_stop_count,
            "random_seed": config["random_seed"],
            "random_state": capture_random_state(
                train_loader_generator
            ),
        },
        last_model_path,
    )

    if should_stop:
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
