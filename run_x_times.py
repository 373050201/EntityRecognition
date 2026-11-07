"""生成 500 张模拟 SCADA 图片和标注"""
import shutil
import subprocess
import sys
from pathlib import Path



GENERATE_COUNT = 500
PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPT_TO_RUN = PROJECT_ROOT / "mock_UI_generator.py"
DATASET_DIR = PROJECT_ROOT / "datasets" / "synthetic"
COUNTER_FILE = PROJECT_ROOT / "counter.txt"
SPLITS = ("synthetic_train", "synthetic_val", "synthetic_test")


def reset_dataset():
    """清空旧合成图片和标签，并重新创建固定的数据集目录。"""
    for split in SPLITS:
        split_dir = DATASET_DIR / "images" / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        annotation_dir = DATASET_DIR / "annotations" / split
        if annotation_dir.exists():
            shutil.rmtree(annotation_dir)
        annotation_dir.mkdir(parents=True, exist_ok=True)

        legacy_annotation_path = DATASET_DIR / "annotations" / f"{split}.json"
        legacy_annotation_path.unlink(missing_ok=True)


reset_dataset()
COUNTER_FILE.unlink(missing_ok=True)
print("旧合成图片和标签已清空，开始生成 500 组数据")

try:
    for index in range(GENERATE_COUNT):
        result = subprocess.run(
            [sys.executable, str(SCRIPT_TO_RUN)],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"第 {index + 1} 次运行失败，退出码：{result.returncode}"
            )
finally:
    COUNTER_FILE.unlink(missing_ok=True)

print("生成完成：synthetic_train=400，synthetic_val=50，synthetic_test=50")
