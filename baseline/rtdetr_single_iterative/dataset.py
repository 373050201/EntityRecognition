"""SCADA实体数据集读取与预处理。"""
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from utils import absolute_xywh_to_normalized_cxcywh



class SCADAEntityDataset(Dataset):
    """读取SCADA图片及其同名实体JSON标注。"""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    REQUIRED_ENTITY_FIELDS = {
        "symbol_class",
        "symbol_bbox",
        "locked",
        "text_bbox",
        "identifier",
    }

    def __init__(
        self,
        dataset_root,
        split="synthetic_train",
        image_size=(540, 960),
        normalize=True,
    ):
        """
        Args:
            dataset_root: 包含 `images/` 和 `annotations/` 的数据集根目录。
            split: `synthetic_train`、`synthetic_val`或`synthetic_test`。
            image_size: 输出 `(height, width)`；设为None时保留原图尺寸。
            normalize: 是否使用ImageNet均值和标准差归一化图像。
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.image_dir = self.dataset_root / "images" / split
        self.annotation_dir = self.dataset_root / "annotations" / split

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"图片目录不存在：{self.image_dir}")
        if not self.annotation_dir.is_dir():
            raise FileNotFoundError(f"标注目录不存在：{self.annotation_dir}")

        image_paths = [
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS
        ]
        self.image_paths = sorted(image_paths, key=self._file_sort_key)
        if not self.image_paths:
            raise RuntimeError(f"图片目录中没有可读取的图片：{self.image_dir}")

        for image_path in self.image_paths:
            annotation_path = self.annotation_dir / f"{image_path.stem}.json"
            if not annotation_path.is_file():
                raise FileNotFoundError(
                    f"图片缺少同名JSON标注：{image_path.name} -> {annotation_path}"
                )

    @staticmethod
    def _file_sort_key(path):
        """数字文件名按数值排序，其他文件名按字符串排序。"""
        try:
            return 0, int(path.stem)
        except ValueError:
            return 1, path.stem

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        annotation_path = self.annotation_dir / f"{image_path.stem}.json"

        with Image.open(image_path) as image:
            image = image.convert("RGB")
        original_width, original_height = image.size

        with open(annotation_path, "r", encoding="utf-8") as f:
            annotation = json.load(f)
        entities = annotation.get("entities")
        if not isinstance(entities, list):
            raise ValueError(f"标注缺少entities列表：{annotation_path}")

        for entity_index, entity in enumerate(entities):
            missing_fields = self.REQUIRED_ENTITY_FIELDS - set(entity)
            if missing_fields:
                raise ValueError(
                    f"{annotation_path} 的第{entity_index}个实体缺少字段："
                    f"{sorted(missing_fields)}"
                )

        labels = torch.tensor(
            [entity["symbol_class"] for entity in entities],
            dtype=torch.long,
        )
        locked = torch.tensor(
            [entity["locked"] for entity in entities],
            dtype=torch.float32,
        )
        identifiers = [str(entity["identifier"]) for entity in entities]

        symbol_boxes = torch.tensor(
            [entity["symbol_bbox"] for entity in entities],
            dtype=torch.float32,
        ).reshape(-1, 4)
        text_boxes = torch.tensor(
            [entity["text_bbox"] for entity in entities],
            dtype=torch.float32,
        ).reshape(-1, 4)

        # JSON绝对像素[x,y,w,h] -> 模型归一化[cx,cy,w,h]。
        symbol_boxes = absolute_xywh_to_normalized_cxcywh(
            symbol_boxes,
            original_width,
            original_height,
        )
        text_boxes = absolute_xywh_to_normalized_cxcywh(
            text_boxes,
            original_width,
            original_height,
        )

        if self.image_size is not None:
            image = TF.resize(image, self.image_size, antialias=True)
        image_tensor = TF.to_tensor(image)
        if self.normalize:
            image_tensor = TF.normalize(
                image_tensor,
                mean=self.IMAGENET_MEAN,
                std=self.IMAGENET_STD,
            )

        target = {
            # N为当前图片中的实体数量，各实体字段保持相同顺序。
            "labels": labels,                    # [N]
            "symbol_boxes": symbol_boxes,       # [N, 4]，归一化cxcywh
            "text_boxes": text_boxes,           # [N, 4]，归一化cxcywh
            "locked": locked,                   # [N]
            "identifiers": identifiers,         # 长度为N的字符串列表
            "original_size": torch.tensor(
                [original_height, original_width],
                dtype=torch.long,
            ),
        }
        return image_tensor, target
