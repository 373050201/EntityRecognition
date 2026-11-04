"""从实体标注构造图元裁剪与锁状态二分类数据集。"""
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from lock_classifier import crop_symbol_region



class LockStateDataset(Dataset):
    """按实体返回带扩边的图元裁剪和二分类标签。"""
    def __init__(self, dataset_root, split, transform, padding_ratio):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.transform = transform
        self.padding_ratio = padding_ratio
        self.samples = []
        annotation_dir = self.dataset_root / "annotations" / split
        for annotation_path in sorted(annotation_dir.glob("*.json")):
            image_path = (
                self.dataset_root
                / "images"
                / split
                / f"{annotation_path.stem}.png"
            )
            with open(annotation_path, "r", encoding="utf-8") as file:
                entities = json.load(file)["entities"]
            for entity in entities:
                x, y, width, height = entity["symbol_bbox"]
                self.samples.append((
                    image_path,
                    (x, y, x + width, y + height),
                    int(bool(entity["locked"])),
                ))
        if not self.samples:
            raise FileNotFoundError(f"未找到锁状态数据划分：{split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, box, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            crop = crop_symbol_region(image, box, self.padding_ratio)
        return self.transform(crop), label

    def class_counts(self):
        """返回未锁和已锁样本数量。"""
        counts = [0, 0]
        for _, _, label in self.samples:
            counts[label] += 1
        return counts

