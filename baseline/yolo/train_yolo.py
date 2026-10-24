"""
YOLO11 目标检测器训练模块：使用预训练权重在 SCADA 数据集上微调。
评估标准：mAP@0.5、mAP@0.5:0.95。
"""
import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO



print("正在加载训练配置...")
yolo_dir = Path(__file__).resolve().parent
os.chdir(yolo_dir)
config_path = yolo_dir / "config_yolo.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

dataset_config = (yolo_dir / config["dataset_config"]).resolve()
pretrained_model = config["pretrained_model"]
cleanup_models = [
    (yolo_dir / model_path).resolve()
    for model_path in config["cleanup_models"]
]
input_size = config["input_size"]
device = config["device"]
epochs = config["epochs"]
batch_size = config["batch_size"]
workers = config["workers"]
patience = config["patience"]
project_dir = (yolo_dir / config["project"]).resolve()
name = config["name"]
output_model = (yolo_dir / config["output_model"]).resolve()

print("正在加载 YOLO11 预训练模型...")
detector = YOLO(pretrained_model)

print("训练开始...")
detector.train(
    data=str(dataset_config),
    epochs=epochs,
    imgsz=input_size,
    batch=batch_size,
    device=device,
    workers=workers,
    patience=patience,
    project=str(project_dir),
    name=name,
    exist_ok=True,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.1,
    scale=0.1,
    fliplr=0.0,
    flipud=0.0,
    mosaic=0.0,
)
print("训练结束")

run_dir = Path(detector.trainer.save_dir).resolve()
best_model = Path(detector.trainer.best).resolve()

output_model.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(best_model, output_model)
print(f"最佳权重已保存至：{output_model}")

shutil.rmtree(run_dir)
cleanup_dir = project_dir
while cleanup_dir != yolo_dir:
    if not cleanup_dir.is_dir() or any(cleanup_dir.iterdir()):
        break
    cleanup_dir.rmdir()
    cleanup_dir = cleanup_dir.parent
print(f"已清理本次训练输出目录：{run_dir}")

for cleanup_model in cleanup_models:
    cleanup_model.unlink(missing_ok=True)
    print(f"已清理训练辅助模型：{cleanup_model}")
