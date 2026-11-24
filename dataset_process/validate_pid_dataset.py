"""独立抽查P&ID切片标注与原始symbols、words和linker的一致性。"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np



PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT_CANDIDATES = [
    PROJECT_ROOT / "tmp" / "p_id_raw",
    PROJECT_ROOT / "datasets" / "p_id",
]
DATASET_ROOT_CANDIDATES = [
    PROJECT_ROOT / "datasets" / "p_id",
    PROJECT_ROOT / "tmp" / "p_id_processed_staging",
]
REPORT_PATH = PROJECT_ROOT / "tmp" / "p_id_validation" / "semantic_audit.json"
SAMPLE_ANNOTATIONS = 2000
RANDOM_SEED = 20261115
TILE_SIZE = 1024


def find_root(candidates, marker):
    """按标志文件定位原始数据或处理后数据。"""
    for candidate in candidates:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"没有找到包含{marker}的目录")


def normalize_xyxy(box):
    """将原始框转换为标准xyxy。"""
    x1, y1, x2, y2 = map(float, box)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def parse_word_references(raw_references):
    """提取linker中的文字引用。"""
    word_ids = []
    for raw_reference in raw_references:
        reference = str(raw_reference)
        if reference.startswith("word_"):
            word_ids.extend(
                part.strip()
                for part in reference.split(",")
                if part.strip()
            )
    return word_ids


def xyxy_to_local_xywh(box, tile_x, tile_y):
    """将整图xyxy框转换为切片内xywh。"""
    x1, y1, x2, y2 = box
    return [x1 - tile_x, y1 - tile_y, x2 - x1, y2 - y1]


def values_equal(left, right, tolerance=1e-6):
    """比较两个数值列表。"""
    return len(left) == len(right) and all(
        abs(float(a) - float(b)) <= tolerance
        for a, b in zip(left, right)
    )


def load_source_records(raw_root, source_id):
    """加载一张原图的三类关系记录。"""
    source_dir = raw_root / str(source_id)
    symbols = np.load(
        source_dir / f"{source_id}_symbols.npy",
        allow_pickle=True,
    )
    words = np.load(
        source_dir / f"{source_id}_words.npy",
        allow_pickle=True,
    )
    linkers = np.load(
        source_dir / f"{source_id}_linker.npy",
        allow_pickle=True,
    )
    return (
        {str(row[0]): row for row in symbols},
        {str(row[0]): row for row in words},
        {str(row[0]): row[1] for row in linkers},
    )


raw_root = find_root(RAW_ROOT_CANDIDATES, "image_2")
dataset_root = find_root(DATASET_ROOT_CANDIDATES, "metadata")
annotation_paths = sorted((dataset_root / "annotations").glob("*/*.json"))
if len(annotation_paths) < SAMPLE_ANNOTATIONS:
    raise RuntimeError("处理后标注数量不足，无法执行预定抽查")

sampled_paths = random.Random(RANDOM_SEED).sample(
    annotation_paths,
    SAMPLE_ANNOTATIONS,
)
source_cache = {}
statistics = Counter()
checked_classes = set()
mismatches = []

for annotation_path in sampled_paths:
    with open(annotation_path, "r", encoding="utf-8") as file:
        annotation = json.load(file)
    source_id = int(annotation["source_image_id"])
    tile_x, tile_y = annotation["tile_bbox"][:2]
    if source_id not in source_cache:
        source_cache[source_id] = load_source_records(raw_root, source_id)
    symbol_map, word_map, linker_map = source_cache[source_id]

    for entity in annotation["entities"]:
        symbol_id = entity["symbol_id"]
        source_symbol = symbol_map.get(symbol_id)
        if source_symbol is None:
            mismatches.append([str(annotation_path), symbol_id, "missing_symbol"])
            continue
        expected_class = int(source_symbol[2]) - 1
        expected_symbol_box = xyxy_to_local_xywh(
            normalize_xyxy(source_symbol[1]),
            tile_x,
            tile_y,
        )
        if int(entity["symbol_class"]) != expected_class:
            mismatches.append([str(annotation_path), symbol_id, "class"])
        if not values_equal(entity["symbol_bbox"], expected_symbol_box):
            mismatches.append([str(annotation_path), symbol_id, "symbol_bbox"])

        expected_texts = []
        for word_id in parse_word_references(linker_map[symbol_id]):
            source_word = word_map[word_id]
            source_word_box = normalize_xyxy(source_word[1])
            source_text = str(source_word[2])
            if (
                source_text
                and source_word_box[2] > source_word_box[0]
                and source_word_box[3] > source_word_box[1]
            ):
                expected_texts.append(
                    {
                        "word_id": word_id,
                        "bbox": xyxy_to_local_xywh(
                            source_word_box,
                            tile_x,
                            tile_y,
                        ),
                        "text": source_text,
                        "orientation": int(source_word[3]),
                    }
                )
        if len(entity["texts"]) != len(expected_texts):
            mismatches.append([str(annotation_path), symbol_id, "text_count"])
        else:
            for actual_text, expected_text in zip(entity["texts"], expected_texts):
                if actual_text["word_id"] != expected_text["word_id"]:
                    mismatches.append([str(annotation_path), symbol_id, "word_id"])
                if actual_text["text"] != expected_text["text"]:
                    mismatches.append([str(annotation_path), symbol_id, "text"])
                if int(actual_text["orientation"]) != expected_text["orientation"]:
                    mismatches.append([str(annotation_path), symbol_id, "orientation"])
                if not values_equal(actual_text["bbox"], expected_text["bbox"]):
                    mismatches.append([str(annotation_path), symbol_id, "text_bbox"])
        checked_classes.add(expected_class)
        statistics.update(
            complete_entities=1,
            text_instances=len(entity["texts"]),
        )

    for entity in annotation["partial_entities"]:
        symbol_id = entity["symbol_id"]
        source_symbol = symbol_map.get(symbol_id)
        if source_symbol is None:
            mismatches.append([str(annotation_path), symbol_id, "missing_partial_symbol"])
            continue
        if int(entity["symbol_class"]) != int(source_symbol[2]) - 1:
            mismatches.append([str(annotation_path), symbol_id, "partial_class"])
        x, y, width, height = map(float, entity["symbol_bbox"])
        if not (
            0 <= x < TILE_SIZE
            and 0 <= y < TILE_SIZE
            and width > 0
            and height > 0
            and x + width <= TILE_SIZE
            and y + height <= TILE_SIZE
        ):
            mismatches.append([str(annotation_path), symbol_id, "partial_bbox"])
        statistics.update(partial_entities=1)

    statistics.update(annotations=1)

report = {
    "random_seed": RANDOM_SEED,
    "sampled_annotations": SAMPLE_ANNOTATIONS,
    "statistics": dict(statistics),
    "checked_class_count": len(checked_classes),
    "checked_classes": sorted(checked_classes),
    "mismatch_count": len(mismatches),
    "mismatches": mismatches[:100],
}
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, ensure_ascii=False, indent=2)

if mismatches:
    raise RuntimeError(f"语义抽查发现{len(mismatches)}项不一致")
print(json.dumps(report, ensure_ascii=False, indent=2))
