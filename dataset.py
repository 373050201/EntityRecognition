"""P&ID层次化实体数据集读取与固定文字槽位预处理。"""
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from utils import absolute_xywh_to_normalized_cxcywh



class PIDEntityDataset(Dataset):
    """读取P&ID切片及其同名层次化实体JSON标注。"""

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    REQUIRED_ENTITY_FIELDS = {"symbol_class", "symbol_bbox", "texts"}

    def __init__(
        self,
        dataset_root,
        split="p_id_train",
        image_size=(1024, 1024),
        normalize=True,
        num_text_slots=3,
        include_partial_entities=True,
    ):
        """
        Args:
            dataset_root: 包含 `images/` 和 `annotations/` 的P&ID数据集根目录。
            split: `p_id_train`、`p_id_val`或`p_id_test`。
            image_size: 输出 `(height, width)`；设为None时保留原图尺寸。
            normalize: 是否使用ImageNet均值和标准差归一化图像。
            num_text_slots: 每个实体在模型输入中填充到的最大文字段数。
            include_partial_entities: 是否保留截断实体的符号监督。
        """
        if num_text_slots <= 0:
            raise ValueError("num_text_slots必须大于0")
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.normalize = normalize
        self.num_text_slots = num_text_slots
        self.include_partial_entities = include_partial_entities
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

        self.annotations = []
        for image_path in self.image_paths:
            annotation_path = self.annotation_dir / f"{image_path.stem}.json"
            if not annotation_path.is_file():
                raise FileNotFoundError(
                    f"图片缺少同名JSON标注：{image_path.name} -> {annotation_path}"
                )
            with open(annotation_path, "r", encoding="utf-8") as file:
                self.annotations.append(json.load(file))

    @staticmethod
    def _file_sort_key(path):
        """切片文件名按源图、横坐标和纵坐标中的数字自然排序。"""
        parts = path.stem.replace("_x", "_").replace("_y", "_").split("_")
        try:
            return 0, tuple(int(part) for part in parts)
        except ValueError:
            return 1, path.stem

    def __len__(self):
        return len(self.image_paths)

    def _build_target(self, annotation, image_width, image_height, annotation_path):
        complete_entities = annotation.get("entities")
        partial_entities = annotation.get("partial_entities", [])
        if not isinstance(complete_entities, list):
            raise ValueError(f"标注缺少entities列表：{annotation_path}")
        if not isinstance(partial_entities, list):
            raise ValueError(f"partial_entities必须为列表：{annotation_path}")

        records = []
        for entity_index, entity in enumerate(complete_entities):
            missing_fields = self.REQUIRED_ENTITY_FIELDS - set(entity)
            if missing_fields:
                raise ValueError(
                    f"{annotation_path} 的第{entity_index}个实体缺少字段："
                    f"{sorted(missing_fields)}"
                )
            texts = entity["texts"]
            if not isinstance(texts, list) or len(texts) > self.num_text_slots:
                raise ValueError(
                    f"{annotation_path} 的实体{entity_index}文字段数量超出"
                    f"0至{self.num_text_slots}范围"
                )
            records.append((entity, texts, True, False))

        if self.include_partial_entities:
            for entity in partial_entities:
                if "symbol_class" not in entity or "symbol_bbox" not in entity:
                    raise ValueError(f"截断实体缺少符号字段：{annotation_path}")
                # 截断实体缺少完整文字位置，只监督图元；所有文字任务统一mask。
                records.append((entity, [], False, True))

        labels = torch.tensor(
            [record[0]["symbol_class"] for record in records],
            dtype=torch.long,
        )
        symbol_boxes = torch.tensor(
            [record[0]["symbol_bbox"] for record in records],
            dtype=torch.float32,
        ).reshape(-1, 4)
        symbol_boxes = absolute_xywh_to_normalized_cxcywh(
            symbol_boxes,
            image_width,
            image_height,
        )

        num_entities = len(records)
        text_boxes = torch.zeros(
            (num_entities, self.num_text_slots, 4),
            dtype=torch.float32,
        )
        text_valid = torch.zeros(
            (num_entities, self.num_text_slots),
            dtype=torch.bool,
        )
        texts = [["" for _ in range(self.num_text_slots)] for _ in records]
        text_supervision = torch.tensor(
            [record[2] for record in records],
            dtype=torch.bool,
        )
        partial = torch.tensor(
            [record[3] for record in records],
            dtype=torch.bool,
        )

        for entity_index, (_, entity_texts, _, _) in enumerate(records):
            for text_index, text_entity in enumerate(entity_texts):
                if "bbox" not in text_entity or "text" not in text_entity:
                    raise ValueError(f"文字子实体缺少bbox或text：{annotation_path}")
                text_boxes[entity_index, text_index] = torch.tensor(
                    text_entity["bbox"],
                    dtype=torch.float32,
                )
                text = str(text_entity["text"])
                texts[entity_index][text_index] = text
                text_valid[entity_index, text_index] = True

        text_boxes = absolute_xywh_to_normalized_cxcywh(
            text_boxes,
            image_width,
            image_height,
        )
        return {
            "labels": labels,                         # [N]
            "symbol_boxes": symbol_boxes,            # [N, 4]
            "text_boxes": text_boxes,                # [N, K, 4]
            "text_valid": text_valid,                # [N, K]
            "text_supervision": text_supervision,    # [N]
            "texts": texts,                          # N个长度为K的字符串列表
            "partial": partial,                      # [N]
            "original_size": torch.tensor(
                [image_height, image_width],
                dtype=torch.long,
            ),
        }

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        annotation_path = self.annotation_dir / f"{image_path.stem}.json"

        with Image.open(image_path) as image:
            image = image.convert("RGB")
        original_width, original_height = image.size
        annotation = self.annotations[index]

        target = self._build_target(
            annotation,
            original_width,
            original_height,
            annotation_path,
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
        return image_tensor, target


# 保留旧名称，避免项目外部脚本导入时立即失效。
SCADAEntityDataset = PIDEntityDataset
