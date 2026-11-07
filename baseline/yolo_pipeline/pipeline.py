"""YOLO11s检测、RapidOCR识别和匈牙利实体关联流水线。"""
import math
import time
import numpy as np
import torch
from rapidocr import RapidOCR
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from lock_classifier import build_lock_classifier, predict_lock_scores



def _point_to_box_distance(symbol_centers, text_boxes):
    """计算每个图元中心到每个文字框最近点的归一化欧氏距离。"""
    symbol_x = symbol_centers[:, 0, None]
    symbol_y = symbol_centers[:, 1, None]
    text_x1 = text_boxes[None, :, 0]
    text_y1 = text_boxes[None, :, 1]
    text_x2 = text_boxes[None, :, 2]
    text_y2 = text_boxes[None, :, 3]
    delta_x = torch.maximum(
        torch.maximum(text_x1 - symbol_x, symbol_x - text_x2),
        torch.zeros_like(text_x1 - symbol_x),
    )
    delta_y = torch.maximum(
        torch.maximum(text_y1 - symbol_y, symbol_y - text_y2),
        torch.zeros_like(text_y1 - symbol_y),
    )
    return torch.sqrt(delta_x.square() + delta_y.square()) / math.sqrt(2)


def _empty_candidates():
    """创建没有任何检测结果时的候选结构。"""
    return {
        "symbol_boxes": torch.empty((0, 4), dtype=torch.float32),
        "symbol_scores": torch.empty((0,), dtype=torch.float32),
        "symbol_labels": torch.empty((0,), dtype=torch.long),
        "lock_scores": torch.empty((0,), dtype=torch.float32),
        "text_boxes": torch.empty((0, 4), dtype=torch.float32),
        "text_scores": torch.empty((0,), dtype=torch.float32),
        "identifiers": [],
        "ocr_scores": torch.empty((0,), dtype=torch.float32),
    }


def associate_candidates(
    candidates,
    symbol_conf_threshold=0.25,
    text_conf_threshold=0.25,
    ocr_conf_threshold=0.5,
    lock_threshold=0.5,
    max_distance=0.06,
):
    """过滤工业流水线候选，并用匈牙利算法完成一对一关联。"""
    symbol_keep = candidates["symbol_scores"] >= symbol_conf_threshold
    text_keep = (
        (candidates["text_scores"] >= text_conf_threshold)
        & (candidates["ocr_scores"] >= ocr_conf_threshold)
    )
    symbol_boxes = candidates["symbol_boxes"][symbol_keep]
    symbol_scores = candidates["symbol_scores"][symbol_keep]
    symbol_labels = candidates["symbol_labels"][symbol_keep]
    lock_scores = candidates["lock_scores"][symbol_keep]
    locked = lock_scores >= lock_threshold
    text_boxes = candidates["text_boxes"][text_keep]
    text_scores = candidates["text_scores"][text_keep]
    ocr_scores = candidates["ocr_scores"][text_keep]
    text_indices = torch.arange(len(candidates["text_scores"]))[text_keep]
    identifiers = [
        candidates["identifiers"][index]
        for index in text_indices.tolist()
    ]

    empty_prediction = {
        "scores": torch.empty((0,), dtype=torch.float32),
        "symbol_boxes": torch.empty((0, 4), dtype=torch.float32),
        "symbol_labels": torch.empty((0,), dtype=torch.long),
        "locked": torch.empty((0,), dtype=torch.bool),
        "text_boxes": torch.empty((0, 4), dtype=torch.float32),
        "identifiers": [],
        "association_distances": torch.empty((0,), dtype=torch.float32),
    }
    if len(symbol_boxes) == 0 or len(text_boxes) == 0:
        return empty_prediction

    symbol_centers = (symbol_boxes[:, :2] + symbol_boxes[:, 2:]) / 2
    distances = _point_to_box_distance(symbol_centers, text_boxes)
    symbol_indices, text_indices = linear_sum_assignment(distances.numpy())
    valid_pairs = [
        (symbol_index, text_index)
        for symbol_index, text_index in zip(symbol_indices, text_indices)
        if distances[symbol_index, text_index] <= max_distance
    ]
    if not valid_pairs:
        return empty_prediction

    matched_symbol_indices = torch.tensor(
        [pair[0] for pair in valid_pairs],
        dtype=torch.long,
    )
    matched_text_indices = torch.tensor(
        [pair[1] for pair in valid_pairs],
        dtype=torch.long,
    )
    matched_scores = torch.minimum(
        symbol_scores[matched_symbol_indices],
        text_scores[matched_text_indices],
    )
    matched_scores = torch.minimum(
        matched_scores,
        ocr_scores[matched_text_indices],
    )
    matched_lock_confidences = torch.where(
        locked[matched_symbol_indices],
        lock_scores[matched_symbol_indices],
        1 - lock_scores[matched_symbol_indices],
    )
    matched_scores = torch.minimum(matched_scores, matched_lock_confidences)
    return {
        "scores": matched_scores,
        "symbol_boxes": symbol_boxes[matched_symbol_indices],
        "symbol_labels": symbol_labels[matched_symbol_indices],
        "locked": locked[matched_symbol_indices],
        "text_boxes": text_boxes[matched_text_indices],
        "identifiers": [
            identifiers[index]
            for index in matched_text_indices.tolist()
        ],
        "association_distances": distances[
            matched_symbol_indices,
            matched_text_indices,
        ],
    }


class YOLOOCRPipeline:
    """封装YOLO11s五类检测、锁分类器和RapidOCR文字识别。"""
    def __init__(self, model_path, lock_checkpoint_path, config):
        self.config = config
        self.detector = YOLO(model_path)
        model_names = self.detector.names
        if isinstance(model_names, dict):
            model_names = [model_names[index] for index in sorted(model_names)]
        if model_names != config["class_names"]:
            raise RuntimeError(
                "YOLO11s权重类别与config_yolo.yaml中的class_names不一致"
            )
        self.lock_device = torch.device(
            f"cuda:{config['device']}"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.lock_classifier = build_lock_classifier(pretrained=False).to(
            self.lock_device
        )
        lock_checkpoint = torch.load(
            lock_checkpoint_path,
            map_location=self.lock_device,
            weights_only=False,
        )
        self.lock_classifier.load_state_dict(
            lock_checkpoint["model_state_dict"],
            strict=True,
        )
        self.lock_classifier.eval()
        self.lock_checkpoint_epoch = lock_checkpoint.get("epoch", "未知")
        self.last_num_symbol_candidates = 0
        self.ocr_engine, self.ocr_provider = self._build_ocr_engine()

    def _build_ocr_engine(self):
        """优先使用配置的CUDA执行器，不可用时回退到CPU。"""
        if self.config["rapidocr_use_cuda"]:
            try:
                engine = RapidOCR(params={
                    "EngineConfig.onnxruntime.providers": [
                        "CUDAExecutionProvider",
                    ],
                    "EngineConfig.onnxruntime.cuda_ep_cfg": {
                        "device_id": self.config["rapidocr_device_id"],
                    },
                })
                return engine, "CUDAExecutionProvider"
            except Exception as error:
                print(f"RapidOCR CUDA初始化失败，回退CPU：{error}")
        return RapidOCR(), "CPUExecutionProvider"

    def _recognize_crop(self, image, box):
        """裁剪单个文字框并返回RapidOCR编号及置信度。"""
        image_width, image_height = image.size
        padding = self.config["ocr_crop_padding"]
        x1, y1, x2, y2 = box
        crop_box = (
            max(int(math.floor(x1)) - padding, 0),
            max(int(math.floor(y1)) - padding, 0),
            min(int(math.ceil(x2)) + padding, image_width),
            min(int(math.ceil(y2)) + padding, image_height),
        )
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            return "", 0.0
        crop = np.asarray(image.crop(crop_box))
        result = self.ocr_engine(
            crop,
            use_det=False,
            use_cls=False,
            use_rec=True,
            text_score=0.0,
        )
        if result is None or not result.txts:
            return "", 0.0
        return str(result.txts[0]).strip(), float(result.scores[0])

    def _synchronize_timing(self):
        """同步CUDA任务，确保分阶段计时覆盖实际执行时间。"""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.lock_device)

    @torch.inference_mode()
    def predict_candidates(
        self,
        image,
        detector_confidence=None,
        stage_timings=None,
    ):
        """执行一次YOLO检测，并对所有Text候选执行RapidOCR。"""
        if detector_confidence is None:
            detector_confidence = self.config["candidate_conf_threshold"]
        if stage_timings is not None:
            self._synchronize_timing()
            stage_start_time = time.perf_counter()
        results = self.detector.predict(
            source=image,
            conf=detector_confidence,
            iou=self.config["nms_iou_threshold"],
            imgsz=self.config["input_size"],
            device=self.config["device"],
            agnostic_nms=self.config["agnostic_nms"],
            max_det=self.config["max_det"],
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self.last_num_symbol_candidates = 0
            if stage_timings is not None:
                self._synchronize_timing()
                stage_timings["yolo"] = (
                    time.perf_counter() - stage_start_time
                )
                stage_timings["lock_classifier"] = 0.0
                stage_timings["rapidocr"] = 0.0
            return _empty_candidates()

        image_width, image_height = image.size
        xyxy = boxes.xyxy.detach().cpu().float()
        scores = boxes.conf.detach().cpu().float()
        classes = boxes.cls.detach().cpu().long()
        scale = torch.tensor(
            [image_width, image_height, image_width, image_height],
            dtype=torch.float32,
        )
        normalized_boxes = (xyxy / scale).clamp(0, 1)
        text_mask = classes == self.config["text_class_id"]
        symbol_mask = ~text_mask
        if stage_timings is not None:
            self._synchronize_timing()
            stage_timings["yolo"] = time.perf_counter() - stage_start_time
            stage_start_time = time.perf_counter()
        lock_scores = predict_lock_scores(
            self.lock_classifier,
            image,
            xyxy[symbol_mask],
            self.config["lock_input_size"],
            self.config["lock_crop_padding_ratio"],
            self.lock_device,
        )
        self.last_num_symbol_candidates = int(symbol_mask.sum())
        if stage_timings is not None:
            self._synchronize_timing()
            stage_timings["lock_classifier"] = (
                time.perf_counter() - stage_start_time
            )
            stage_start_time = time.perf_counter()
        identifiers = []
        ocr_scores = []
        for box in xyxy[text_mask].tolist():
            identifier, ocr_score = self._recognize_crop(image, box)
            identifiers.append(identifier)
            ocr_scores.append(ocr_score)
        if stage_timings is not None:
            self._synchronize_timing()
            stage_timings["rapidocr"] = (
                time.perf_counter() - stage_start_time
            )
        return {
            "symbol_boxes": normalized_boxes[symbol_mask],
            "symbol_scores": scores[symbol_mask],
            "symbol_labels": classes[symbol_mask],
            "lock_scores": lock_scores,
            "text_boxes": normalized_boxes[text_mask],
            "text_scores": scores[text_mask],
            "identifiers": identifiers,
            "ocr_scores": torch.tensor(ocr_scores, dtype=torch.float32),
        }

    def predict(self, image, stage_timings=None):
        """执行工业流水线完整推理并返回组装后的实体。"""
        detector_confidence = min(
            self.config["symbol_conf_threshold"],
            self.config["text_conf_threshold"],
        )
        candidates = self.predict_candidates(
            image,
            detector_confidence,
            stage_timings=stage_timings,
        )
        if stage_timings is not None:
            self._synchronize_timing()
            stage_start_time = time.perf_counter()
        prediction = associate_candidates(
            candidates,
            symbol_conf_threshold=self.config["symbol_conf_threshold"],
            text_conf_threshold=self.config["text_conf_threshold"],
            ocr_conf_threshold=self.config["ocr_conf_threshold"],
            lock_threshold=self.config["lock_threshold"],
            max_distance=self.config["association_max_distance"],
        )
        if stage_timings is not None:
            self._synchronize_timing()
            stage_timings["association"] = (
                time.perf_counter() - stage_start_time
            )
        return prediction
