"""确定性保留P&ID训练集困难空切片，并将其余空切片移至tmp备份。"""
import json
import shutil
from collections import defaultdict
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "p_id"
SPLIT = "p_id_train"
TARGET_EMPTY_TILES = 1500
MIN_EMPTY_TILES_PER_SOURCE = 3

IMAGE_DIR = DATASET_ROOT / "images" / SPLIT
ANNOTATION_DIR = DATASET_ROOT / "annotations" / SPLIT
METADATA_PATH = DATASET_ROOT / "metadata" / "statistics.json"
PRUNING_MANIFEST_PATH = (
    DATASET_ROOT / "metadata" / "empty_tile_pruning.json"
)
BACKUP_ROOT = PROJECT_ROOT / "tmp" / "p_id_pruned_empty_tiles"
BACKUP_IMAGE_DIR = BACKUP_ROOT / "images" / SPLIT
BACKUP_ANNOTATION_DIR = BACKUP_ROOT / "annotations" / SPLIT


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def ensure_within_directory(path, directory):
    resolved_path = path.resolve()
    resolved_directory = directory.resolve()
    if resolved_path != resolved_directory and resolved_directory not in (
        resolved_path.parents
    ):
        raise RuntimeError(f"路径超出预期目录：{resolved_path}")


def find_image_path(stem):
    candidates = [
        IMAGE_DIR / f"{stem}.jpg",
        IMAGE_DIR / f"{stem}.jpeg",
        IMAGE_DIR / f"{stem}.png",
    ]
    image_paths = [path for path in candidates if path.is_file()]
    if len(image_paths) != 1:
        raise RuntimeError(f"无法唯一确定切片图片：{stem}")
    return image_paths[0]


ensure_within_directory(DATASET_ROOT, PROJECT_ROOT)
ensure_within_directory(BACKUP_ROOT, PROJECT_ROOT / "tmp")
if not IMAGE_DIR.is_dir() or not ANNOTATION_DIR.is_dir():
    raise FileNotFoundError("P&ID训练集图片或标注目录不存在")
if PRUNING_MANIFEST_PATH.is_file():
    existing_manifest = read_json(PRUNING_MANIFEST_PATH)
    remaining_images = len(list(IMAGE_DIR.glob("*")))
    remaining_annotations = len(list(ANNOTATION_DIR.glob("*.json")))
    expected_tiles = int(existing_manifest["remaining_train_tiles"])
    if (
        remaining_images != expected_tiles
        or remaining_annotations != expected_tiles
    ):
        raise RuntimeError("清理清单与当前训练集文件数量不一致")
    print(
        f"训练集已经完成空切片清理：{expected_tiles}张，"
        "无需重复执行"
    )
    raise SystemExit(0)

empty_tiles = []
for annotation_path in sorted(ANNOTATION_DIR.glob("*.json")):
    annotation = read_json(annotation_path)
    if annotation.get("entities") or annotation.get("partial_entities"):
        continue
    image_path = find_image_path(annotation_path.stem)
    empty_tiles.append(
        {
            "stem": annotation_path.stem,
            "source_image_id": int(annotation["source_image_id"]),
            "image_bytes": image_path.stat().st_size,
            "image_path": image_path,
            "annotation_path": annotation_path,
        }
    )

if len(empty_tiles) < TARGET_EMPTY_TILES:
    raise RuntimeError(
        f"当前空切片只有{len(empty_tiles)}个，少于目标保留数"
        f"{TARGET_EMPTY_TILES}，可能已经执行过清理"
    )

tiles_by_source = defaultdict(list)
for tile in empty_tiles:
    tiles_by_source[tile["source_image_id"]].append(tile)

# 相同分辨率和JPEG质量下，文件越大通常表示线条、表格或文字越丰富。
# 每张源图先保留至少三个困难负样本，再按文件大小全局补足目标数量。
kept_stems = set()
for source_tiles in tiles_by_source.values():
    source_tiles.sort(key=lambda item: (-item["image_bytes"], item["stem"]))
    kept_stems.update(
        tile["stem"]
        for tile in source_tiles[:MIN_EMPTY_TILES_PER_SOURCE]
    )

remaining_tiles = sorted(
    (tile for tile in empty_tiles if tile["stem"] not in kept_stems),
    key=lambda item: (-item["image_bytes"], item["stem"]),
)
for tile in remaining_tiles:
    if len(kept_stems) >= TARGET_EMPTY_TILES:
        break
    kept_stems.add(tile["stem"])

if len(kept_stems) != TARGET_EMPTY_TILES:
    raise RuntimeError(
        f"困难负样本选择数量错误：{len(kept_stems)}"
    )

removed_tiles = [
    tile for tile in empty_tiles if tile["stem"] not in kept_stems
]
for tile in removed_tiles:
    backup_image_path = BACKUP_IMAGE_DIR / tile["image_path"].name
    backup_annotation_path = (
        BACKUP_ANNOTATION_DIR / tile["annotation_path"].name
    )
    if backup_image_path.exists() or backup_annotation_path.exists():
        raise FileExistsError(
            f"备份目标已存在，拒绝覆盖：{tile['stem']}"
        )

BACKUP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
for tile in removed_tiles:
    shutil.move(
        str(tile["image_path"]),
        str(BACKUP_IMAGE_DIR / tile["image_path"].name),
    )
    shutil.move(
        str(tile["annotation_path"]),
        str(BACKUP_ANNOTATION_DIR / tile["annotation_path"].name),
    )

statistics = read_json(METADATA_PATH)
train_statistics = statistics["splits"][SPLIT]
original_train_tiles = int(train_statistics["tiles"])
original_empty_tiles = int(train_statistics["empty_tiles"])
train_statistics["tiles"] = original_train_tiles - len(removed_tiles)
train_statistics["empty_tiles"] = len(kept_stems)
statistics["overall"]["tiles"] -= len(removed_tiles)
write_json(METADATA_PATH, statistics)

manifest = {
    "split": SPLIT,
    "selection_method": (
        "每张源图按JPEG文件大小保留至少3个空切片，"
        "再按文件大小全局补足到1500个"
    ),
    "original_empty_tiles": len(empty_tiles),
    "kept_empty_tiles": len(kept_stems),
    "moved_empty_tiles": len(removed_tiles),
    "original_train_tiles": original_train_tiles,
    "remaining_train_tiles": train_statistics["tiles"],
    "backup_root": str(BACKUP_ROOT.relative_to(PROJECT_ROOT)),
    "kept": [
        {
            "stem": tile["stem"],
            "source_image_id": tile["source_image_id"],
            "image_bytes": tile["image_bytes"],
        }
        for tile in sorted(empty_tiles, key=lambda item: item["stem"])
        if tile["stem"] in kept_stems
    ],
    "moved": [
        {
            "stem": tile["stem"],
            "source_image_id": tile["source_image_id"],
            "image_bytes": tile["image_bytes"],
        }
        for tile in sorted(removed_tiles, key=lambda item: item["stem"])
    ],
}
write_json(PRUNING_MANIFEST_PATH, manifest)
write_json(BACKUP_ROOT / "manifest.json", manifest)

print(
    f"训练空切片：{len(empty_tiles)} -> {len(kept_stems)}，"
    f"移至tmp备份：{len(removed_tiles)}"
)
print(
    f"训练切片：{original_train_tiles} -> "
    f"{train_statistics['tiles']}"
)
