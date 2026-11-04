"""MobileNetV3-Small锁状态二分类器及图元裁剪工具。"""
import math
import torch
from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)
from torchvision.transforms import v2



IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_lock_classifier(pretrained=False):
    """构建替换二分类头后约1.52M参数的MobileNetV3-Small。"""
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    return model


def build_lock_transform(input_size, training=False):
    """创建锁状态训练增强或确定性推理预处理。"""
    transforms = [v2.ToImage()]
    if training:
        transforms.extend([
            v2.RandomAffine(
                degrees=5,
                translate=(0.06, 0.06),
                scale=(0.90, 1.10),
                fill=255,
            ),
            v2.ColorJitter(brightness=0.10, contrast=0.10),
        ])
    transforms.extend([
        v2.Resize((input_size, input_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return v2.Compose(transforms)


def crop_symbol_region(image, box, padding_ratio):
    """按预测框尺寸比例扩边并裁剪图元区域。"""
    image_width, image_height = image.size
    x1, y1, x2, y2 = box
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    padding_x = width * padding_ratio
    padding_y = height * padding_ratio
    crop_box = (
        max(int(math.floor(x1 - padding_x)), 0),
        max(int(math.floor(y1 - padding_y)), 0),
        min(int(math.ceil(x2 + padding_x)), image_width),
        min(int(math.ceil(y2 + padding_y)), image_height),
    )
    return image.crop(crop_box)


@torch.inference_mode()
def predict_lock_scores(
    model,
    image,
    boxes,
    input_size,
    padding_ratio,
    device,
):
    """批量裁剪YOLO图元预测框并返回已锁类别概率。"""
    if len(boxes) == 0:
        return torch.empty((0,), dtype=torch.float32)
    transform = build_lock_transform(input_size, training=False)
    crops = [
        transform(crop_symbol_region(image, box, padding_ratio))
        for box in boxes.tolist()
    ]
    inputs = torch.stack(crops).to(device, non_blocking=device.type == "cuda")
    return model(inputs).softmax(dim=-1)[:, 1].detach().cpu()
