"""读取工业流水线评估所需的原图与完整实体标注。"""
import json
from pathlib import Path
from PIL import Image
import torch



def absolute_xywh_to_normalized_cxcywh(boxes, image_width, image_height):
    """把绝对像素xywh转换为归一化cxcywh。"""
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    x, y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            (x + width / 2) / image_width,
            (y + height / 2) / image_height,
            width / image_width,
            height / image_height,
        ),
        dim=-1,
    )


class SCADAIndustrialDataset:
    """按图片返回PIL原图和严格实体评估标注。"""
    def __init__(self, dataset_root, split):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.annotation_paths = sorted(
            (self.dataset_root / "annotations" / split).glob("*.json")
        )
        if not self.annotation_paths:
            raise FileNotFoundError(f"未找到数据划分：{split}")

    def __len__(self):
        return len(self.annotation_paths)

    def __getitem__(self, index):
        annotation_path = self.annotation_paths[index]
        image_path = (
            self.dataset_root
            / "images"
            / self.split
            / f"{annotation_path.stem}.png"
        )
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        with open(annotation_path, "r", encoding="utf-8") as file:
            entities = json.load(file)["entities"]
        target = {
            "symbol_boxes": absolute_xywh_to_normalized_cxcywh(
                [entity["symbol_bbox"] for entity in entities],
                image_width,
                image_height,
            ),
            "text_boxes": absolute_xywh_to_normalized_cxcywh(
                [entity["text_bbox"] for entity in entities],
                image_width,
                image_height,
            ),
            "symbol_labels": torch.tensor(
                [entity["symbol_class"] for entity in entities],
                dtype=torch.long,
            ),
            "locked": torch.tensor(
                [bool(entity["locked"]) for entity in entities],
                dtype=torch.bool,
            ),
            "identifiers": [str(entity["identifier"]) for entity in entities],
            "original_size": (image_height, image_width),
            "image_path": image_path,
        }
        return image, target

