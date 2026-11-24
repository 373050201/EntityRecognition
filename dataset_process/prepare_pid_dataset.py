"""将原始Dataset-P&ID转换为层次化实体切片数据集。"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont



PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT_CANDIDATES = [
    PROJECT_ROOT / "tmp" / "p_id_raw",
    PROJECT_ROOT / "datasets" / "p_id",
]
OUTPUT_ROOT = PROJECT_ROOT / "tmp" / "p_id_processed_staging"
VALIDATION_ROOT = PROJECT_ROOT / "tmp" / "p_id_validation"

SHEET_COUNT = 500
TILE_SIZE = 1024
TILE_STRIDE = 768
SPLIT_SEED = 42
SPLIT_SIZES = {
    "p_id_train": 400,
    "p_id_val": 50,
    "p_id_test": 50,
}
JPEG_QUALITY = 95
VALIDATION_SAMPLE_COUNT = 12


def find_raw_root():
    """定位仍保持原始发布结构的P&ID目录。"""
    for candidate in RAW_ROOT_CANDIDATES:
        if (
            (candidate / "image_2").is_dir()
            and (candidate / "0" / "0_symbols.npy").is_file()
        ):
            return candidate
    raise FileNotFoundError("没有找到原始P&ID目录")


def normalize_xyxy(box):
    """将坐标方向不统一的原始框转换为标准xyxy。"""
    x1, y1, x2, y2 = map(float, box)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def xyxy_to_xywh(box):
    """将xyxy框转换为绝对像素xywh。"""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def local_xywh(box, tile_x, tile_y):
    """将整图xyxy框转换为切片内绝对像素xywh。"""
    x1, y1, x2, y2 = box
    return [x1 - tile_x, y1 - tile_y, x2 - x1, y2 - y1]


def clip_local_xywh(box, tile_x, tile_y):
    """裁剪整图框并转换为切片内绝对像素xywh。"""
    x1, y1, x2, y2 = box
    clipped = [
        max(x1, tile_x),
        max(y1, tile_y),
        min(x2, tile_x + TILE_SIZE),
        min(y2, tile_y + TILE_SIZE),
    ]
    return local_xywh(clipped, tile_x, tile_y)


def build_positions(length):
    """生成覆盖完整图像且末端贴边的切片起点。"""
    if length <= TILE_SIZE:
        return [0]
    positions = list(range(0, length - TILE_SIZE + 1, TILE_STRIDE))
    final_position = length - TILE_SIZE
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def parse_word_references(raw_references):
    """从linker记录中提取可能以逗号连接的word引用。"""
    word_ids = []
    for raw_reference in raw_references:
        reference = str(raw_reference)
        if not reference.startswith("word_"):
            continue
        word_ids.extend(
            part.strip()
            for part in reference.split(",")
            if part.strip()
        )
    return word_ids


def load_source_entities(raw_root, source_id):
    """读取一张原图并恢复符号及其可变长度文字集合。"""
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

    word_map = {str(row[0]): row for row in words}
    linker_map = {str(row[0]): row[1] for row in linkers}
    entities = []
    for symbol_id, raw_symbol_box, raw_class_id in symbols:
        symbol_id = str(symbol_id)
        symbol_box = normalize_xyxy(raw_symbol_box)
        texts = []
        for word_id in parse_word_references(linker_map[symbol_id]):
            word = word_map[word_id]
            word_box = normalize_xyxy(word[1])
            text = str(word[2])
            if not text or word_box[2] <= word_box[0] or word_box[3] <= word_box[1]:
                continue
            texts.append(
                {
                    "word_id": word_id,
                    "box": word_box,
                    "text": text,
                    "orientation": int(word[3]),
                }
            )

        all_boxes = [symbol_box] + [text["box"] for text in texts]
        entity_box = [
            min(box[0] for box in all_boxes),
            min(box[1] for box in all_boxes),
            max(box[2] for box in all_boxes),
            max(box[3] for box in all_boxes),
        ]
        entities.append(
            {
                "entity_id": f"{source_id}:{symbol_id}",
                "symbol_id": symbol_id,
                "symbol_class": int(raw_class_id) - 1,
                "symbol_class_original": int(raw_class_id),
                "symbol_box": symbol_box,
                "entity_box": entity_box,
                "texts": texts,
            }
        )
    return entities


def contains_box(tile_box, box):
    """判断一个框是否完整位于切片中。"""
    return (
        tile_box[0] <= box[0]
        and tile_box[1] <= box[1]
        and box[2] <= tile_box[2]
        and box[3] <= tile_box[3]
    )


def contains_symbol_center(tile_box, symbol_box):
    """判断符号中心是否位于切片中。"""
    center_x = (symbol_box[0] + symbol_box[2]) / 2
    center_y = (symbol_box[1] + symbol_box[3]) / 2
    return (
        tile_box[0] <= center_x < tile_box[2]
        and tile_box[1] <= center_y < tile_box[3]
    )


def convert_complete_entity(entity, tile_x, tile_y):
    """生成一个完整的切片级层次化实体标注。"""
    return {
        "entity_id": entity["entity_id"],
        "symbol_id": entity["symbol_id"],
        "symbol_class": entity["symbol_class"],
        "symbol_bbox": local_xywh(entity["symbol_box"], tile_x, tile_y),
        "texts": [
            {
                "word_id": text["word_id"],
                "bbox": local_xywh(text["box"], tile_x, tile_y),
                "text": text["text"],
                "orientation": text["orientation"],
            }
            for text in entity["texts"]
        ],
    }


def convert_partial_entity(entity, tile_x, tile_y):
    """记录只能用于部分监督的边界实体。"""
    return {
        "entity_id": entity["entity_id"],
        "symbol_id": entity["symbol_id"],
        "symbol_class": entity["symbol_class"],
        "symbol_bbox": clip_local_xywh(
            entity["symbol_box"],
            tile_x,
            tile_y,
        ),
        "text_count": len(entity["texts"]),
        "ignore_text": True,
        "reason": "entity_crosses_tile_boundary",
    }


def build_splits():
    """使用固定随机种子生成400/50/50整图划分。"""
    source_ids = list(range(SHEET_COUNT))
    random.Random(SPLIT_SEED).shuffle(source_ids)
    splits = {}
    offset = 0
    for split_name, split_size in SPLIT_SIZES.items():
        splits[split_name] = sorted(source_ids[offset:offset + split_size])
        offset += split_size
    if offset != SHEET_COUNT:
        raise ValueError("数据划分数量与完整图纸总数不一致")
    return splits


def make_output_directories():
    """创建与合成数据一致的图片、标注和元数据目录。"""
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"临时输出目录已经存在：{OUTPUT_ROOT}")
    for split_name in SPLIT_SIZES:
        (OUTPUT_ROOT / "images" / split_name).mkdir(parents=True)
        (OUTPUT_ROOT / "annotations" / split_name).mkdir(parents=True)
    (OUTPUT_ROOT / "metadata").mkdir(parents=True)


def write_json(path, value):
    """以统一格式写入UTF-8 JSON。"""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def process_dataset(raw_root, splits):
    """切分全部原图并生成层次化实体JSON。"""
    source_to_split = {
        source_id: split_name
        for split_name, source_ids in splits.items()
        for source_id in source_ids
    }
    statistics = Counter()
    characters = set()
    entity_complete_counts = Counter()
    split_statistics = {
        split_name: Counter()
        for split_name in SPLIT_SIZES
    }

    for source_id in range(SHEET_COUNT):
        split_name = source_to_split[source_id]
        image_path = raw_root / "image_2" / f"{source_id}.jpg"
        source_entities = load_source_entities(raw_root, source_id)
        with Image.open(image_path) as source_image:
            source_image = source_image.convert("RGB")
            width, height = source_image.size
            x_positions = build_positions(width)
            y_positions = build_positions(height)

            for tile_y in y_positions:
                for tile_x in x_positions:
                    tile_box = [
                        tile_x,
                        tile_y,
                        tile_x + TILE_SIZE,
                        tile_y + TILE_SIZE,
                    ]
                    complete_entities = []
                    partial_entities = []
                    for entity in source_entities:
                        if contains_box(tile_box, entity["entity_box"]):
                            converted = convert_complete_entity(
                                entity,
                                tile_x,
                                tile_y,
                            )
                            complete_entities.append(converted)
                            entity_complete_counts[entity["entity_id"]] += 1
                            for text in converted["texts"]:
                                characters.update(text["text"])
                        elif contains_symbol_center(tile_box, entity["symbol_box"]):
                            partial_entities.append(
                                convert_partial_entity(
                                    entity,
                                    tile_x,
                                    tile_y,
                                )
                            )

                    tile_name = (
                        f"{source_id:03d}_x{tile_x:04d}_y{tile_y:04d}"
                    )
                    tile_image = source_image.crop(
                        (
                            tile_x,
                            tile_y,
                            tile_x + TILE_SIZE,
                            tile_y + TILE_SIZE,
                        )
                    )
                    tile_image.save(
                        OUTPUT_ROOT / "images" / split_name / f"{tile_name}.jpg",
                        quality=JPEG_QUALITY,
                        subsampling=0,
                    )
                    annotation = {
                        "image_id": tile_name,
                        "source_image_id": source_id,
                        "source_file": f"{source_id}.jpg",
                        "tile_bbox": [tile_x, tile_y, TILE_SIZE, TILE_SIZE],
                        "image_size": [TILE_SIZE, TILE_SIZE],
                        "entities": complete_entities,
                        "partial_entities": partial_entities,
                    }
                    write_json(
                        OUTPUT_ROOT
                        / "annotations"
                        / split_name
                        / f"{tile_name}.json",
                        annotation,
                    )

                    statistics.update(
                        tiles=1,
                        complete_entity_instances=len(complete_entities),
                        partial_entity_instances=len(partial_entities),
                    )
                    split_statistics[split_name].update(
                        tiles=1,
                        complete_entity_instances=len(complete_entities),
                        partial_entity_instances=len(partial_entities),
                        empty_tiles=not complete_entities and not partial_entities,
                    )

        statistics.update(source_images=1, source_entities=len(source_entities))
        split_statistics[split_name].update(
            source_images=1,
            source_entities=len(source_entities),
        )
        print(
            f"[{source_id + 1:03d}/{SHEET_COUNT}] "
            f"{source_id}.jpg -> {split_name}",
            flush=True,
        )

    missing_entities = [
        entity_id
        for source_id in range(SHEET_COUNT)
        for entity_id in (
            entity["entity_id"]
            for entity in load_source_entities(raw_root, source_id)
        )
        if entity_complete_counts[entity_id] == 0
    ]
    if missing_entities:
        raise RuntimeError(
            f"有{len(missing_entities)}个实体没有完整落入任何切片"
        )
    statistics["unique_complete_entities"] = len(entity_complete_counts)
    return statistics, split_statistics, sorted(characters)


def validate_box(box):
    """检查切片内xywh框是否合法。"""
    x, y, width, height = map(float, box)
    return (
        0 <= x < TILE_SIZE
        and 0 <= y < TILE_SIZE
        and width > 0
        and height > 0
        and x + width <= TILE_SIZE
        and y + height <= TILE_SIZE
    )


def validate_processed_dataset(splits):
    """检查图片、标注、框、类别、文本槽位和划分完整性。"""
    seen_sources = {}
    unique_entities = set()
    totals = Counter()
    for split_name, source_ids in splits.items():
        image_dir = OUTPUT_ROOT / "images" / split_name
        annotation_dir = OUTPUT_ROOT / "annotations" / split_name
        image_paths = sorted(image_dir.glob("*.jpg"))
        annotation_paths = sorted(annotation_dir.glob("*.json"))
        if len(image_paths) != len(annotation_paths):
            raise RuntimeError(f"{split_name}的图片和标注数量不一致")

        for image_path, annotation_path in zip(image_paths, annotation_paths):
            if image_path.stem != annotation_path.stem:
                raise RuntimeError(f"图片与标注不同名：{image_path.name}")
            with open(annotation_path, "r", encoding="utf-8") as file:
                annotation = json.load(file)
            source_id = int(annotation["source_image_id"])
            if source_id not in source_ids:
                raise RuntimeError(f"{annotation_path}被写入错误划分")
            previous_split = seen_sources.setdefault(source_id, split_name)
            if previous_split != split_name:
                raise RuntimeError(f"原图{source_id}跨越多个数据划分")

            with Image.open(image_path) as image:
                if image.size != (TILE_SIZE, TILE_SIZE):
                    raise RuntimeError(f"切片尺寸错误：{image_path}")

            for entity in annotation["entities"]:
                if not 0 <= int(entity["symbol_class"]) < 32:
                    raise RuntimeError(f"符号类别越界：{annotation_path}")
                if not validate_box(entity["symbol_bbox"]):
                    raise RuntimeError(f"符号框越界：{annotation_path}")
                if len(entity["texts"]) > 3:
                    raise RuntimeError(f"文字槽位超过3个：{annotation_path}")
                for text in entity["texts"]:
                    if not text["text"] or not validate_box(text["bbox"]):
                        raise RuntimeError(f"文字标注无效：{annotation_path}")
                unique_entities.add(entity["entity_id"])
                totals.update(
                    complete_entity_instances=1,
                    text_instances=len(entity["texts"]),
                )
            for entity in annotation["partial_entities"]:
                if not validate_box(entity["symbol_bbox"]):
                    raise RuntimeError(f"边界符号框无效：{annotation_path}")
                totals.update(partial_entity_instances=1)
            totals.update(tiles=1)

    if len(seen_sources) != SHEET_COUNT:
        raise RuntimeError("处理结果没有覆盖全部原图")
    if len(unique_entities) != 59498:
        raise RuntimeError(
            f"完整实体覆盖数量错误：{len(unique_entities)} != 59498"
        )
    return totals


def draw_xywh(draw, box, color, width=3):
    """在验证图中绘制xywh框。"""
    x, y, box_width, box_height = box
    draw.rectangle(
        [x, y, x + box_width, y + box_height],
        outline=color,
        width=width,
    )


def create_validation_samples():
    """抽取多文字、无文字和普通实体切片并绘制标注。"""
    if VALIDATION_ROOT.exists():
        shutil.rmtree(VALIDATION_ROOT)
    VALIDATION_ROOT.mkdir(parents=True)

    candidates = []
    for split_name in SPLIT_SIZES:
        annotation_dir = OUTPUT_ROOT / "annotations" / split_name
        for annotation_path in annotation_dir.glob("*.json"):
            with open(annotation_path, "r", encoding="utf-8") as file:
                annotation = json.load(file)
            text_counts = [
                len(entity["texts"])
                for entity in annotation["entities"]
            ]
            priority = (
                max(text_counts, default=-1),
                any(count == 0 for count in text_counts),
                len(annotation["entities"]),
            )
            if annotation["entities"]:
                candidates.append((priority, split_name, annotation_path))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    selected_sources = set()
    for _, split_name, annotation_path in candidates:
        source_id = int(annotation_path.stem.split("_")[0])
        if source_id in selected_sources:
            continue
        selected.append((split_name, annotation_path))
        selected_sources.add(source_id)
        if len(selected) == VALIDATION_SAMPLE_COUNT:
            break

    font = ImageFont.load_default()
    rendered_paths = []
    for sample_index, (split_name, annotation_path) in enumerate(selected):
        with open(annotation_path, "r", encoding="utf-8") as file:
            annotation = json.load(file)
        image_path = (
            OUTPUT_ROOT
            / "images"
            / split_name
            / f"{annotation_path.stem}.jpg"
        )
        with Image.open(image_path) as image:
            rendered = image.convert("RGB")
        draw = ImageDraw.Draw(rendered)
        for entity in annotation["entities"]:
            draw_xywh(draw, entity["symbol_bbox"], "red", width=4)
            symbol_x, symbol_y = entity["symbol_bbox"][:2]
            label = (
                f"C{entity['symbol_class']:02d} "
                f"{entity['symbol_id']}"
            )
            draw.text(
                (symbol_x, max(0, symbol_y - 14)),
                label,
                fill="red",
                font=font,
                stroke_width=2,
                stroke_fill="white",
            )
            for text in entity["texts"]:
                draw_xywh(draw, text["bbox"], "lime", width=3)
                text_x, text_y = text["bbox"][:2]
                draw.line(
                    [
                        symbol_x,
                        symbol_y,
                        text_x,
                        text_y,
                    ],
                    fill="cyan",
                    width=2,
                )
                draw.text(
                    (text_x, max(0, text_y - 13)),
                    text["text"],
                    fill="lime",
                    font=font,
                    stroke_width=2,
                    stroke_fill="black",
                )
        for entity in annotation["partial_entities"]:
            draw_xywh(draw, entity["symbol_bbox"], "orange", width=3)

        output_path = VALIDATION_ROOT / (
            f"sample_{sample_index:02d}_{annotation_path.stem}.jpg"
        )
        rendered.save(output_path, quality=95, subsampling=0)
        rendered_paths.append(output_path)

    thumbnails = []
    for path in rendered_paths:
        with Image.open(path) as image:
            thumbnail = image.convert("RGB")
        thumbnail.thumbnail((512, 512))
        thumbnails.append(thumbnail)
    columns = 3
    rows = (len(thumbnails) + columns - 1) // columns
    contact_sheet = Image.new(
        "RGB",
        (columns * 512, rows * 512),
        color="white",
    )
    for index, thumbnail in enumerate(thumbnails):
        contact_sheet.paste(
            thumbnail,
            ((index % columns) * 512, (index // columns) * 512),
        )
    contact_sheet.save(VALIDATION_ROOT / "contact_sheet.jpg", quality=95)
    return rendered_paths


def write_metadata(raw_root, splits, statistics, split_statistics, characters):
    """保存划分、字符集、统计量和可复现参数。"""
    split_payload = {
        "random_seed": SPLIT_SEED,
        "splits": splits,
    }
    write_json(OUTPUT_ROOT / "metadata" / "splits.json", split_payload)

    source_hashes = {}
    for source_id in range(SHEET_COUNT):
        image_path = raw_root / "image_2" / f"{source_id}.jpg"
        source_hashes[str(source_id)] = hashlib.sha256(
            image_path.read_bytes()
        ).hexdigest()
    write_json(
        OUTPUT_ROOT / "metadata" / "source_image_sha256.json",
        source_hashes,
    )
    write_json(
        OUTPUT_ROOT / "metadata" / "characters.json",
        {
            "characters": characters,
            "character_string": "".join(characters),
        },
    )
    write_json(
        OUTPUT_ROOT / "metadata" / "statistics.json",
        {
            "overall": dict(statistics),
            "splits": {
                split_name: dict(counts)
                for split_name, counts in split_statistics.items()
            },
        },
    )

    data_yaml = {
        "path": "..",
        "train": "images/p_id_train",
        "val": "images/p_id_val",
        "test": "images/p_id_test",
        "num_classes": 32,
        "class_names": [
            f"class_{class_id:02d}"
            for class_id in range(1, 33)
        ],
        "annotation_format": "hierarchical_entity_json",
        "box_format": "absolute_xywh",
        "tile_size": [TILE_SIZE, TILE_SIZE],
        "tile_stride": TILE_STRIDE,
    }
    with open(
        OUTPUT_ROOT / "metadata" / "data.yaml",
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            data_yaml,
            file,
            allow_unicode=True,
            sort_keys=False,
        )


raw_root = find_raw_root()
print(f"原始数据：{raw_root}")
print(f"临时输出：{OUTPUT_ROOT}")
make_output_directories()
dataset_splits = build_splits()
processing_statistics, per_split_statistics, associated_characters = (
    process_dataset(raw_root, dataset_splits)
)
validation_statistics = validate_processed_dataset(dataset_splits)
write_metadata(
    raw_root,
    dataset_splits,
    processing_statistics,
    per_split_statistics,
    associated_characters,
)
validation_samples = create_validation_samples()
print("处理统计：", dict(processing_statistics))
print("验证统计：", dict(validation_statistics))
print("关联字符集：", repr("".join(associated_characters)))
print(f"可视化抽样：{len(validation_samples)}张")
print("P&ID切片数据集生成完成")
