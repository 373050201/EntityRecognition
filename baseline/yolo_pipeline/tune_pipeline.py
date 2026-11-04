"""在验证集上搜索YOLO11s与RapidOCR工业流水线参数。"""
import itertools
import re
from pathlib import Path
import yaml
from dataset import SCADAIndustrialDataset
from pipeline import YOLOOCRPipeline, associate_candidates
from utils import calculate_pipeline_entity_metrics, summarize_counts



coarse_symbol_confidences = [0.10, 0.20, 0.30, 0.40, 0.50]
coarse_text_confidences = [0.10, 0.20, 0.30, 0.40, 0.50]
coarse_ocr_confidences = [0.00, 0.30, 0.50, 0.70, 0.90]
coarse_lock_thresholds = [0.10, 0.30, 0.50, 0.70, 0.90]
coarse_max_distances = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
fine_detector_radius = 0.04
fine_detector_step = 0.02
fine_ocr_radius = 0.05
fine_ocr_step = 0.05
fine_lock_radius = 0.10
fine_lock_step = 0.05
fine_distance_radius = 0.01
fine_distance_step = 0.005


def make_local_values(center, radius, step, lower, upper):
    """生成以粗搜索最优值为中心的去重细搜索序列。"""
    start = max(lower, center - radius)
    end = min(upper, center + radius)
    count = int(round((end - start) / step))
    return sorted({
        round(start + index * step, 6)
        for index in range(count + 1)
    } | {round(center, 6)})


def evaluate_parameters(
    cached_samples,
    config,
    symbol_confidence,
    text_confidence,
    ocr_confidence,
    lock_threshold,
    max_distance,
):
    """在缓存的验证候选上计算一组工业流水线参数。"""
    counts = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }
    for candidates, target in cached_samples:
        prediction = associate_candidates(
            candidates,
            symbol_conf_threshold=symbol_confidence,
            text_conf_threshold=text_confidence,
            ocr_conf_threshold=ocr_confidence,
            lock_threshold=lock_threshold,
            max_distance=max_distance,
        )
        metrics = calculate_pipeline_entity_metrics(
            [prediction],
            [target],
            symbol_iou_threshold=config["symbol_iou_threshold"],
            text_iou_threshold=config["text_iou_threshold"],
        )
        for name in counts:
            counts[name] += metrics[name]
    return summarize_counts(**counts)


def search_parameters(
    cached_samples,
    config,
    symbol_confidences,
    text_confidences,
    ocr_confidences,
    lock_thresholds,
    max_distances,
    stage_name,
):
    """遍历候选组合，以实体F1为首要标准返回最佳参数。"""
    combinations = list(itertools.product(
        symbol_confidences,
        text_confidences,
        ocr_confidences,
        lock_thresholds,
        max_distances,
    ))
    best_result = None
    print(f"{stage_name}：共{len(combinations)}组参数")
    for index, (
        symbol_confidence,
        text_confidence,
        ocr_confidence,
        lock_threshold,
        max_distance,
    ) in enumerate(combinations, start=1):
        metrics = evaluate_parameters(
            cached_samples,
            config,
            symbol_confidence,
            text_confidence,
            ocr_confidence,
            lock_threshold,
            max_distance,
        )
        result = {
            "symbol_conf_threshold": symbol_confidence,
            "text_conf_threshold": text_confidence,
            "ocr_conf_threshold": ocr_confidence,
            "lock_threshold": lock_threshold,
            "association_max_distance": max_distance,
            "metrics": metrics,
        }
        result_key = (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            metrics["accuracy"],
            -max_distance,
            symbol_confidence,
            text_confidence,
            ocr_confidence,
            lock_threshold,
        )
        if best_result is None or result_key > best_result["key"]:
            result["key"] = result_key
            best_result = result
        if index % 100 == 0 or index == len(combinations):
            print(
                f"  已完成 {index}/{len(combinations)}，"
                f"当前最佳实体F1 {best_result['metrics']['f1']:.4f}"
            )
    return best_result


def update_config_file(config_path, best_result):
    """保留原配置格式，将验证集选出的五个参数写回配置文件。"""
    config_text = config_path.read_text(encoding="utf-8")
    for name in (
        "symbol_conf_threshold",
        "text_conf_threshold",
        "ocr_conf_threshold",
        "lock_threshold",
        "association_max_distance",
    ):
        value = best_result[name]
        config_text, replacement_count = re.subn(
            rf"(?m)^{name}:\s*[^#\r\n]+",
            f"{name}: {value:.6g}",
            config_text,
            count=1,
        )
        if replacement_count != 1:
            raise RuntimeError(f"config_yolo.yaml中未唯一找到参数：{name}")
    config_path.write_text(config_text, encoding="utf-8")


yolo_dir = Path(__file__).resolve().parent
config_path = yolo_dir / "config_yolo.yaml"
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
dataset = SCADAIndustrialDataset(
    yolo_dir / config["dataset_root"],
    config["val_split"],
)
model_path = yolo_dir / config["model_path"]
lock_checkpoint_path = yolo_dir / config["lock_best_checkpoint"]
if not model_path.is_file():
    raise FileNotFoundError(f"YOLO11s检查点不存在：{model_path}")
if not lock_checkpoint_path.is_file():
    raise FileNotFoundError(f"锁状态分类器不存在：{lock_checkpoint_path}")
pipeline = YOLOOCRPipeline(model_path, lock_checkpoint_path, config)
print(f"参数搜索数据集：{config['val_split']}，共{len(dataset)}张图片")
print(f"RapidOCR执行器：{pipeline.ocr_provider}")

# 三个模块只执行一次，后续搜索复用完整候选缓存。
cached_samples = []
print("开始缓存验证集检测与OCR结果...")
for index in range(len(dataset)):
    image, target = dataset[index]
    candidates = pipeline.predict_candidates(
        image,
        config["candidate_conf_threshold"],
    )
    cached_samples.append((candidates, target))
    if (index + 1) % 10 == 0 or index + 1 == len(dataset):
        print(f"  已缓存 {index + 1}/{len(dataset)}")

coarse_best = search_parameters(
    cached_samples,
    config,
    coarse_symbol_confidences,
    coarse_text_confidences,
    coarse_ocr_confidences,
    coarse_lock_thresholds,
    coarse_max_distances,
    "粗搜索",
)
fine_symbol_confidences = make_local_values(
    coarse_best["symbol_conf_threshold"],
    fine_detector_radius,
    fine_detector_step,
    config["candidate_conf_threshold"],
    0.99,
)
fine_text_confidences = make_local_values(
    coarse_best["text_conf_threshold"],
    fine_detector_radius,
    fine_detector_step,
    config["candidate_conf_threshold"],
    0.99,
)
fine_ocr_confidences = make_local_values(
    coarse_best["ocr_conf_threshold"],
    fine_ocr_radius,
    fine_ocr_step,
    0.0,
    1.0,
)
fine_lock_thresholds = make_local_values(
    coarse_best["lock_threshold"],
    fine_lock_radius,
    fine_lock_step,
    0.0,
    1.0,
)
fine_max_distances = make_local_values(
    coarse_best["association_max_distance"],
    fine_distance_radius,
    fine_distance_step,
    0.0,
    1.0,
)
best_result = search_parameters(
    cached_samples,
    config,
    fine_symbol_confidences,
    fine_text_confidences,
    fine_ocr_confidences,
    fine_lock_thresholds,
    fine_max_distances,
    "细搜索",
)
update_config_file(config_path, best_result)

best_metrics = best_result["metrics"]
print("验证集参数搜索完成，最佳参数已写入config_yolo.yaml：")
print(
    f"symbol_conf_threshold: {best_result['symbol_conf_threshold']:.4f}\n"
    f"text_conf_threshold: {best_result['text_conf_threshold']:.4f}\n"
    f"ocr_conf_threshold: {best_result['ocr_conf_threshold']:.4f}\n"
    f"lock_threshold: {best_result['lock_threshold']:.4f}\n"
    f"association_max_distance: {best_result['association_max_distance']:.4f}"
)
print("验证实体指标 | F1 | Precision | Recall | 严格准确率")
print(
    "             | "
    f"{best_metrics['f1']:.4f} | "
    f"{best_metrics['precision']:.4f} | "
    f"{best_metrics['recall']:.4f} | "
    f"{best_metrics['accuracy']:.4f}"
)
print(
    "验证实体计数 | "
    f"TP {best_metrics['true_positive']} | "
    f"FP {best_metrics['false_positive']} | "
    f"FN {best_metrics['false_negative']}"
)
