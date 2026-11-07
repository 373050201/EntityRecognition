"""在测试集上评估YOLO11s与RapidOCR工业实体流水线。"""
import time
from pathlib import Path
import torch
import yaml
from ultralytics.utils.torch_utils import get_flops
from dataset import SCADAIndustrialDataset
from pipeline import YOLOOCRPipeline
from utils import (
    calculate_pipeline_entity_metrics,
    create_entity_breakdown_counts,
    summarize_counts,
    summarize_entity_breakdown,
)



def measure_pipeline_efficiency(pipeline, dataset, config):
    """统计完整流水线端到端效率，并单独执行分阶段耗时剖析。"""
    first_image, _ = dataset[0]
    warmup_iterations = config["efficiency_warmup_iterations"]
    print(f"预热完整工业流水线：{warmup_iterations}次")
    for _ in range(warmup_iterations):
        pipeline.predict(first_image)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    elapsed_seconds = 0.0
    total_lock_crops = 0
    for index in range(len(dataset)):
        image, _ = dataset[index]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        pipeline.predict(image)
        total_lock_crops += pipeline.last_num_symbol_candidates
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_seconds += time.perf_counter() - start_time
    peak_memory_gib = (
        torch.cuda.max_memory_allocated() / 1024 ** 3
        if torch.cuda.is_available()
        else 0.0
    )

    stage_seconds = {
        "yolo": 0.0,
        "lock_classifier": 0.0,
        "rapidocr": 0.0,
        "association": 0.0,
    }
    print("开始分阶段计时：逐阶段同步执行一次测试集")
    for index in range(len(dataset)):
        image, _ = dataset[index]
        image_stage_timings = {}
        pipeline.predict(image, stage_timings=image_stage_timings)
        for name in stage_seconds:
            stage_seconds[name] += image_stage_timings[name]
    stage_total_seconds = sum(stage_seconds.values())

    detector_gflops = get_flops(
        pipeline.detector.model,
        imgsz=config["input_size"],
    )
    lock_gflops_per_crop = get_flops(
        pipeline.lock_classifier,
        imgsz=config["lock_input_size"],
    )
    average_lock_crops = total_lock_crops / len(dataset)
    return {
        "fps": len(dataset) / elapsed_seconds,
        "estimated_gflops_per_image": (
            detector_gflops + lock_gflops_per_crop * average_lock_crops
        ),
        "detector_gflops_per_image": detector_gflops,
        "lock_gflops_per_crop": lock_gflops_per_crop,
        "average_lock_crops": average_lock_crops,
        "pytorch_peak_memory_gib": peak_memory_gib,
        "stage_total_fps": len(dataset) / stage_total_seconds,
        "stage_milliseconds_per_image": {
            name: seconds / len(dataset) * 1000
            for name, seconds in stage_seconds.items()
        },
        "stage_percentages": {
            name: seconds / stage_total_seconds * 100
            for name, seconds in stage_seconds.items()
        },
    }


yolo_dir = Path(__file__).resolve().parent
with open(yolo_dir / "config_yolo.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)
dataset = SCADAIndustrialDataset(
    yolo_dir / config["dataset_root"],
    config["test_split"],
)
model_path = yolo_dir / config["model_path"]
lock_checkpoint_path = yolo_dir / config["lock_best_checkpoint"]
if not model_path.is_file():
    raise FileNotFoundError(f"YOLO11s检查点不存在：{model_path}")
if not lock_checkpoint_path.is_file():
    raise FileNotFoundError(f"锁状态分类器不存在：{lock_checkpoint_path}")
pipeline = YOLOOCRPipeline(model_path, lock_checkpoint_path, config)
print(f"评估数据集：{config['test_split']}，共{len(dataset)}张图片")
print(f"RapidOCR执行器：{pipeline.ocr_provider}")
print(f"锁状态分类器轮次：{pipeline.lock_checkpoint_epoch}")
print(
    "固定参数："
    f"图元置信度{config['symbol_conf_threshold']}，"
    f"文字置信度{config['text_conf_threshold']}，"
    f"OCR置信度{config['ocr_conf_threshold']}，"
    f"锁分类阈值{config['lock_threshold']}，"
    f"匹配距离{config['association_max_distance']}"
)

print("开始统计完整工业流水线效率...")
efficiency_metrics = measure_pipeline_efficiency(pipeline, dataset, config)

entity_counts = {
    "true_positive": 0,
    "false_positive": 0,
    "false_negative": 0,
}
breakdown_counts = create_entity_breakdown_counts()
print("开始评估完整实体...")
for index in range(len(dataset)):
    image, target = dataset[index]
    prediction = pipeline.predict(image)
    batch_metrics = calculate_pipeline_entity_metrics(
        [prediction],
        [target],
        symbol_iou_threshold=config["symbol_iou_threshold"],
        text_iou_threshold=config["text_iou_threshold"],
    )
    for name in entity_counts:
        entity_counts[name] += batch_metrics[name]
    for name in breakdown_counts:
        breakdown_counts[name] += batch_metrics["breakdown_counts"][name]

entity_metrics = summarize_counts(**entity_counts)
entity_breakdown = summarize_entity_breakdown(breakdown_counts)
print("实体指标   | F1 | Precision | Recall | 严格准确率")
print(
    "           | "
    f"{entity_metrics['f1']:.4f} | "
    f"{entity_metrics['precision']:.4f} | "
    f"{entity_metrics['recall']:.4f} | "
    f"{entity_metrics['accuracy']:.4f}"
)
print(
    "实体计数   | "
    f"TP {entity_metrics['true_positive']} | "
    f"FP {entity_metrics['false_positive']} | "
    f"FN {entity_metrics['false_negative']}"
)
print("拆解指标   | 图IoU | 图框 | 文IoU | 文框 | 图类 | 锁 | 字符 | 编号")
print(
    "           | "
    f"{entity_breakdown['symbol_mean_iou']:.4f} | "
    f"{entity_breakdown['symbol_box_accuracy']:.4f} | "
    f"{entity_breakdown['text_mean_iou']:.4f} | "
    f"{entity_breakdown['text_box_accuracy']:.4f} | "
    f"{entity_breakdown['symbol_class_accuracy']:.4f} | "
    f"{entity_breakdown['locked_accuracy']:.4f} | "
    f"{entity_breakdown['identifier_character_accuracy']:.4f} | "
    f"{entity_breakdown['identifier_exact_accuracy']:.4f}"
)
print("效率指标   | FPS | 估算GFLOPs/图 | PyTorch峰值显存(GiB)")
print(
    "           | "
    f"{efficiency_metrics['fps']:.2f} | "
    f"{efficiency_metrics['estimated_gflops_per_image']:.2f} | "
    f"{efficiency_metrics['pytorch_peak_memory_gib']:.3f}"
)
print(
    "FLOPs拆解  | "
    f"YOLO11s {efficiency_metrics['detector_gflops_per_image']:.2f} GFLOPs/图 | "
    f"锁分类器 {efficiency_metrics['lock_gflops_per_crop']:.4f} GFLOPs/裁剪"
)
print("阶段耗时   | YOLO11s | 锁分类器 | RapidOCR | 匹配与组装")
print(
    "毫秒/图    | "
    f"{efficiency_metrics['stage_milliseconds_per_image']['yolo']:.2f} | "
    f"{efficiency_metrics['stage_milliseconds_per_image']['lock_classifier']:.2f} | "
    f"{efficiency_metrics['stage_milliseconds_per_image']['rapidocr']:.2f} | "
    f"{efficiency_metrics['stage_milliseconds_per_image']['association']:.2f}"
)
print(
    "耗时占比   | "
    f"{efficiency_metrics['stage_percentages']['yolo']:.2f}% | "
    f"{efficiency_metrics['stage_percentages']['lock_classifier']:.2f}% | "
    f"{efficiency_metrics['stage_percentages']['rapidocr']:.2f}% | "
    f"{efficiency_metrics['stage_percentages']['association']:.2f}%"
)
print(
    "说明：FPS包含YOLO11s、图元裁剪、锁状态分类、文字裁剪、"
    "RapidOCR和匈牙利匹配；估算FLOPs为YOLO11s加平均图元候选数乘以"
    f"单次锁分类FLOPs（平均{efficiency_metrics['average_lock_crops']:.2f}次/图），"
    "不包含RapidOCR的ONNX运算；"
    "峰值显存仅统计PyTorch分配，不包含ONNX Runtime的RapidOCR显存。"
)
print(
    "分阶段计时说明：为保证GPU与ONNX异步任务归入正确阶段，各阶段边界均执行"
    "CUDA同步，因此分阶段耗时总和对应的"
    f"{efficiency_metrics['stage_total_fps']:.2f} FPS仅用于瓶颈分析，"
    "不替代上方无分段同步的端到端FPS。"
)
