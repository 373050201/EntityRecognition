"""把实体JSON转换为YOLO11s五类检测标签。"""
import json
import os
import shutil
from pathlib import Path
from PIL import Image
import yaml



def convert_box(box, image_width, image_height):
    """把绝对像素xywh转换为YOLO归一化cxcywh。"""
    x, y, width, height = box
    return (
        (x + width / 2) / image_width,
        (y + height / 2) / image_height,
        width / image_width,
        height / image_height,
    )


def prepare_yolo_dataset(yolo_dir, config):
    """为训练、验证和测试划分生成四类图元加Text标签。"""
    dataset_root = (yolo_dir / config["dataset_root"]).resolve()
    output_root = yolo_dir / "yolo_dataset"
    text_class_id = config["text_class_id"]
    total_images = 0
    total_labels = 0
    for split_name in (
        config["train_split"],
        config["val_split"],
        config["test_split"],
    ):
        annotation_dir = dataset_root / "annotations" / split_name
        image_dir = dataset_root / "images" / split_name
        output_image_dir = output_root / "images" / split_name
        label_dir = output_root / "labels" / split_name
        output_image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        annotation_paths = sorted(annotation_dir.glob("*.json"))
        expected_image_names = set()
        expected_label_names = set()
        for annotation_path in annotation_paths:
            image_path = image_dir / f"{annotation_path.stem}.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"图像不存在：{image_path}")
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            with open(annotation_path, "r", encoding="utf-8") as file:
                annotation = json.load(file)

            output_image_path = output_image_dir / image_path.name
            if not output_image_path.exists():
                try:
                    os.link(image_path, output_image_path)
                except OSError:
                    shutil.copy2(image_path, output_image_path)

            label_lines = []
            for entity in annotation["entities"]:
                symbol_class = int(entity["symbol_class"])
                symbol_box = convert_box(
                    entity["symbol_bbox"],
                    image_width,
                    image_height,
                )
                text_box = convert_box(
                    entity["text_bbox"],
                    image_width,
                    image_height,
                )
                label_lines.append(
                    f"{symbol_class} "
                    + " ".join(f"{value:.8f}" for value in symbol_box)
                )
                label_lines.append(
                    f"{text_class_id} "
                    + " ".join(f"{value:.8f}" for value in text_box)
                )

            label_path = label_dir / f"{annotation_path.stem}.txt"
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            expected_image_names.add(output_image_path.name)
            expected_label_names.add(label_path.name)
            total_images += 1
            total_labels += len(label_lines)

        for old_image_path in output_image_dir.glob("*.png"):
            if old_image_path.name not in expected_image_names:
                old_image_path.unlink()
        for old_label_path in label_dir.glob("*.txt"):
            if old_label_path.name not in expected_label_names:
                old_label_path.unlink()
        print(
            f"{split_name}：{len(annotation_paths)}张图片，"
            f"{sum(1 for _ in label_dir.glob('*.txt'))}个标签文件"
        )
    print(f"YOLO数据准备完成：{total_images}张图片，{total_labels}个检测框")


if __name__ == "__main__":
    yolo_dir = Path(__file__).resolve().parent
    with open(yolo_dir / "config_yolo.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    prepare_yolo_dataset(yolo_dir, config)
