"""
项目通用工具函数
"""
import random
from queue import Queue
from threading import Thread
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou



CTC_CHARACTERS = ' "()-0123456789ABCDEFGHIJKLMNOPQRSTUVWXZx°'
CTC_CHARACTER_TO_INDEX = {
    character: index for index, character in enumerate(CTC_CHARACTERS)
}
CTC_CHARACTER_SET = frozenset(CTC_CHARACTERS)
CTC_BLANK_INDEX = len(CTC_CHARACTERS)
CTC_NUM_CLASSES = len(CTC_CHARACTERS) + 1



def set_random_seed(seed):
    """统一固定各类随机源，同时保留GPU高性能算法。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 当前模型的CTC、ROIAlign和可变形采样没有确定性CUDA实现。
    # 只固定随机源，不强制慢速确定性算法，并允许cuDNN自动选择高效实现。
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)


def seed_data_loader_worker(_worker_id):
    """让每个DataLoader worker从PyTorch分配的种子派生自己的随机状态。"""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class BackgroundPrefetchLoader:
    """使用后台线程预取CPU批次，使图片处理与GPU计算重叠。"""

    def __init__(self, data_loader, prefetch_batches=2):
        if prefetch_batches <= 0:
            raise ValueError("预取批次数必须大于0")
        self.data_loader = data_loader
        self.prefetch_batches = prefetch_batches

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        queue = Queue(maxsize=self.prefetch_batches)
        sentinel = object()

        def load_batches():
            try:
                for batch in self.data_loader:
                    queue.put((batch, None))
            except BaseException as error:
                queue.put((None, error))
            finally:
                queue.put(sentinel)

        thread = Thread(target=load_batches, daemon=True)
        thread.start()
        while True:
            item = queue.get()
            if item is sentinel:
                break
            batch, error = item
            if error is not None:
                raise error
            yield batch


def capture_random_state(data_loader_generator):
    """保存断点续训所需的全部随机数状态。"""
    random_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "data_loader_generator": data_loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        random_state["cuda"] = torch.cuda.get_rng_state_all()
    return random_state


def restore_random_state(random_state, data_loader_generator):
    """恢复随机数状态，使断点后的下一轮沿用原训练随机序列。"""
    random.setstate(random_state["python"])
    np.random.set_state(random_state["numpy"])
    # torch.load(map_location="cuda")会连随机状态一起移到GPU，
    # 但CPU默认生成器和DataLoader生成器只接受CPU ByteTensor。
    torch.set_rng_state(random_state["torch"].cpu())
    data_loader_generator.set_state(
        random_state["data_loader_generator"].cpu()
    )
    if torch.cuda.is_available() and "cuda" in random_state:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in random_state["cuda"]]
        )


def _prepare_boxes(boxes):
    """将输入转换为浮点 Tensor，并检查最后一维是否为4。"""
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if boxes.ndim == 0 or boxes.shape[-1] != 4:
        raise ValueError("bbox 的 shape 必须为 [4] 或 [..., 4]")
    return boxes


def _check_image_size(image_width, image_height):
    """检查图像宽高，避免归一化时除以零。"""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width 和 image_height 必须大于0")


def absolute_xywh_to_normalized_cxcywh(boxes, image_width, image_height):
    """
    将绝对像素 `[x, y, w, h]` 转换为归一化 `[cx, cy, w, h]`。

    Args:
        boxes: shape 为 `[4]` 或 `[N, 4]` 的 bbox。
        image_width: 当前图像宽度。
        image_height: 当前图像高度。

    Returns:
        与输入 shape 相同的 float32 Tensor，坐标通常位于 `[0, 1]`。
    """
    _check_image_size(image_width, image_height)
    boxes = _prepare_boxes(boxes)

    x, y, width, height = boxes.unbind(-1)
    normalized_boxes = torch.stack(
        (
            (x + width / 2) / image_width,
            (y + height / 2) / image_height,
            width / image_width,
            height / image_height,
        ),
        dim=-1,
    )
    return normalized_boxes


def normalized_cxcywh_to_absolute_xywh(boxes, image_width, image_height):
    """
    将归一化 `[cx, cy, w, h]` 还原为绝对像素 `[x, y, w, h]`。

    该函数主要用于把模型预测结果还原到原图坐标，方便可视化和导出。
    """
    _check_image_size(image_width, image_height)
    boxes = _prepare_boxes(boxes)

    center_x, center_y, width, height = boxes.unbind(-1)
    absolute_width = width * image_width
    absolute_height = height * image_height
    absolute_boxes = torch.stack(
        (
            center_x * image_width - absolute_width / 2,
            center_y * image_height - absolute_height / 2,
            absolute_width,
            absolute_height,
        ),
        dim=-1,
    )
    return absolute_boxes


def cxcywh_to_xyxy(boxes):
    """将 `[cx, cy, w, h]` 转换为计算 GIoU 所需的 `[x1, y1, x2, y2]`。"""
    boxes = _prepare_boxes(boxes)
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )


def encode_texts(texts, device=None):
    """将 P&ID 文字列表编码为 CTCLoss 使用的一维目标和长度。"""
    encoded = []
    lengths = []
    for text in texts:
        text = str(text)
        if not text:
            raise ValueError("CTC 文字不能为空")
        unknown_characters = sorted(set(text) - CTC_CHARACTER_SET)
        if unknown_characters:
            raise ValueError(
                f"文字包含字符集之外的字符：{unknown_characters!r}，实际文字为：{text!r}"
            )
        encoded.extend(CTC_CHARACTER_TO_INDEX[character] for character in text)
        lengths.append(len(text))

    return (
        torch.tensor(encoded, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def ctc_greedy_decode(text_logits, blank_index=CTC_BLANK_INDEX):
    """
    将 `[B, Q, K, T, C]` 的字符 logits 贪心解码为三维字符串列表。

    解码顺序是先合并相邻重复类别，再删除 CTC blank。
    """
    if text_logits.ndim != 5:
        raise ValueError("text_logits的shape必须为[B, Q, K, T, C]")
    if not 0 <= blank_index < text_logits.shape[-1]:
        raise ValueError("blank_index 超出字符类别范围")

    predicted_indices = text_logits.argmax(dim=-1).detach().cpu()
    decoded_batch = []
    for image_predictions in predicted_indices:
        decoded_queries = []
        for query_predictions in image_predictions:
            decoded_slots = []
            for sequence in query_predictions:
                previous_index = None
                characters = []
                for character_index in sequence.tolist():
                    if (
                        character_index != previous_index
                        and character_index != blank_index
                    ):
                        if character_index >= len(CTC_CHARACTERS):
                            raise ValueError("预测字符索引超出当前 P&ID 字符集")
                        characters.append(CTC_CHARACTERS[character_index])
                    previous_index = character_index
                decoded_slots.append("".join(characters))
            decoded_queries.append(decoded_slots)
        decoded_batch.append(decoded_queries)
    return decoded_batch


def summarize_entity_metrics(
    true_positive,
    false_positive,
    false_negative,
):
    """根据整个验证集累计的实体计数计算准确率、精确率、召回率和F1。"""
    accuracy_denominator = true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    entity_accuracy = (
        true_positive / accuracy_denominator
        if accuracy_denominator > 0 else 0.0
    )
    entity_precision = (
        true_positive / precision_denominator
        if precision_denominator > 0 else 0.0
    )
    entity_recall = (
        true_positive / recall_denominator
        if recall_denominator > 0 else 0.0
    )
    entity_f1 = (
        2 * entity_precision * entity_recall
        / (entity_precision + entity_recall)
        if entity_precision + entity_recall > 0 else 0.0
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "entity_accuracy": entity_accuracy,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
    }


def create_entity_breakdown_counts():
    """创建可跨 batch 累加的实体拆解计数器。"""
    return {
        "num_targets": 0,
        "num_texts": 0,
        "num_characters": 0,
        "symbol_iou_sum": 0.0,
        "text_iou_sum": 0.0,
        "symbol_box_correct": 0,
        "text_box_correct": 0,
        "text_count_correct": 0,
        "symbol_class_correct": 0,
        "text_character_correct": 0,
        "text_exact_correct": 0,
    }


def summarize_entity_breakdown(counts):
    """将验证集累计的子任务计数转换为便于诊断的比例指标。"""
    num_targets = counts["num_targets"]
    num_texts = counts["num_texts"]
    num_characters = counts["num_characters"]

    def entity_rate(name):
        return counts[name] / num_targets if num_targets > 0 else 0.0

    return {
        "symbol_mean_iou": entity_rate("symbol_iou_sum"),
        "symbol_box_accuracy": entity_rate("symbol_box_correct"),
        "text_mean_iou": (
            counts["text_iou_sum"] / num_texts if num_texts > 0 else 0.0
        ),
        "text_box_accuracy": (
            counts["text_box_correct"] / num_texts if num_texts > 0 else 0.0
        ),
        "text_count_accuracy": entity_rate("text_count_correct"),
        "symbol_class_accuracy": entity_rate("symbol_class_correct"),
        "text_character_accuracy": (
            counts["text_character_correct"] / num_characters
            if num_characters > 0 else 0.0
        ),
        "text_exact_accuracy": (
            counts["text_exact_correct"] / num_texts if num_texts > 0 else 0.0
        ),
    }


def _levenshtein_distance(prediction, target):
    """计算两个编号字符串的编辑距离，用于统计字符级准确率。"""
    previous_row = list(range(len(target) + 1))
    for prediction_index, prediction_character in enumerate(prediction, start=1):
        current_row = [prediction_index]
        for target_index, target_character in enumerate(target, start=1):
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[target_index] + 1,
                    previous_row[target_index - 1]
                    + (prediction_character != target_character),
                )
            )
        previous_row = current_row
    return previous_row[-1]


@torch.no_grad()
def calculate_entity_metrics(
    outputs,
    targets,
    entity_conf_threshold=0.5,
    symbol_iou_threshold=0.5,
    text_iou_threshold=0.5,
    text_presence_threshold=0.5,
    decoded_texts=None,
):
    """按图元外层匹配和文字内层集合匹配执行严格实体评估。"""
    thresholds = (
        entity_conf_threshold,
        symbol_iou_threshold,
        text_iou_threshold,
        text_presence_threshold,
    )
    if any(threshold < 0 or threshold > 1 for threshold in thresholds):
        raise ValueError("实体评估的所有阈值都必须位于[0, 1]")

    batch_size, num_queries = outputs["pred_logits"].shape[:2]
    if len(targets) != batch_size:
        raise ValueError("targets数量必须与batch size一致")

    if decoded_texts is None:
        decoded_texts = ctc_greedy_decode(outputs["pred_text_logits"])
    if len(decoded_texts) != batch_size:
        raise ValueError("decoded_texts数量必须与batch size一致")
    true_positive = 0
    false_positive = 0
    false_negative = 0
    breakdown_counts = create_entity_breakdown_counts()

    for batch_index in range(batch_size):
        foreground_scores, foreground_labels = outputs["pred_logits"][
            batch_index
        ].sigmoid().max(-1)
        keep = foreground_scores >= entity_conf_threshold
        kept_query_indices = torch.arange(
            num_queries,
            device=keep.device,
        )[keep]

        predicted_labels = foreground_labels[keep]
        predicted_presence = (
            outputs["pred_text_presence"][batch_index, keep].sigmoid()
            >= text_presence_threshold
        )
        predicted_symbol_boxes = cxcywh_to_xyxy(
            outputs["pred_symbol_boxes"][batch_index, keep]
        ).clamp(0, 1)
        predicted_text_boxes = outputs["pred_text_boxes"][batch_index, keep]
        predicted_texts = [
            decoded_texts[batch_index][query_index]
            for query_index in kept_query_indices.tolist()
        ]

        target_labels = targets[batch_index]["labels"].to(predicted_labels.device)
        target_symbol_boxes = cxcywh_to_xyxy(
            targets[batch_index]["symbol_boxes"].to(predicted_labels.device)
        ).clamp(0, 1)
        target_text_boxes = targets[batch_index]["text_boxes"].to(
            predicted_labels.device
        )
        target_text_valid = targets[batch_index]["text_valid"].to(
            predicted_labels.device
        )
        target_text_supervision = targets[batch_index]["text_supervision"].to(
            predicted_labels.device
        )
        target_texts = targets[batch_index]["texts"]

        num_predictions = len(predicted_labels)
        num_targets = len(target_labels)
        image_true_positive = 0
        complete_target_count = int(target_text_supervision.sum().item())
        breakdown_counts["num_targets"] += complete_target_count
        for target_index in torch.where(target_text_supervision)[0].tolist():
            for text_index in torch.where(target_text_valid[target_index])[0].tolist():
                target_text = target_texts[target_index][text_index]
                breakdown_counts["num_texts"] += 1
                breakdown_counts["num_characters"] += len(target_text)

        if num_predictions > 0 and num_targets > 0:
            symbol_ious = box_iou(predicted_symbol_boxes, target_symbol_boxes)
            valid_symbol_pairs = symbol_ious >= symbol_iou_threshold
            matching_cost = torch.where(
                valid_symbol_pairs,
                1 - symbol_ious,
                1e3 + (1 - symbol_ious),
            )
            prediction_indices, target_indices = linear_sum_assignment(
                matching_cost.cpu().numpy()
            )

            for prediction_index, target_index in zip(
                prediction_indices,
                target_indices,
            ):
                symbol_iou = symbol_ious[prediction_index, target_index].item()
                if not bool(target_text_supervision[target_index]):
                    # 边界截断实体不参与完整实体指标，匹配到它的预测也不计误检。
                    continue
                symbol_box_correct = bool(
                    valid_symbol_pairs[prediction_index, target_index]
                )
                symbol_class_correct = bool(
                    predicted_labels[prediction_index]
                    == target_labels[target_index]
                )

                active_slots = torch.where(predicted_presence[prediction_index])[0]
                valid_text_indices = torch.where(target_text_valid[target_index])[0]
                text_count_correct = len(active_slots) == len(valid_text_indices)
                all_text_boxes_correct = text_count_correct
                all_texts_correct = text_count_correct
                if len(active_slots) > 0 and len(valid_text_indices) > 0:
                    source_text_boxes = cxcywh_to_xyxy(
                        predicted_text_boxes[prediction_index, active_slots]
                    ).clamp(0, 1)
                    target_boxes = cxcywh_to_xyxy(
                        target_text_boxes[target_index, valid_text_indices]
                    ).clamp(0, 1)
                    text_ious = box_iou(source_text_boxes, target_boxes)
                    source_indices, local_target_indices = linear_sum_assignment(
                        (1 - text_ious).cpu().numpy()
                    )
                    matched_text_ious = []
                    for source_index, local_target_index in zip(
                        source_indices,
                        local_target_indices,
                    ):
                        text_iou = text_ious[
                            source_index,
                            local_target_index,
                        ].item()
                        matched_text_ious.append(text_iou)
                        breakdown_counts["text_iou_sum"] += text_iou
                        text_box_correct = text_iou >= text_iou_threshold
                        breakdown_counts["text_box_correct"] += text_box_correct
                        slot_index = int(active_slots[source_index])
                        text_index = int(valid_text_indices[local_target_index])
                        predicted_text = predicted_texts[prediction_index][slot_index]
                        target_text = target_texts[target_index][text_index]
                        text_exact_correct = predicted_text == target_text
                        breakdown_counts["text_exact_correct"] += text_exact_correct
                        text_distance = _levenshtein_distance(
                            predicted_text,
                            target_text,
                        )
                        breakdown_counts["text_character_correct"] += max(
                            len(target_text) - text_distance,
                            0,
                        )
                        all_texts_correct = all_texts_correct and text_exact_correct
                    all_text_boxes_correct = (
                        all_text_boxes_correct
                        and len(matched_text_ious) == len(valid_text_indices)
                        and all(
                            iou >= text_iou_threshold
                            for iou in matched_text_ious
                        )
                    )
                elif len(valid_text_indices) > 0:
                    all_text_boxes_correct = False
                    all_texts_correct = False

                breakdown_counts["symbol_iou_sum"] += symbol_iou
                breakdown_counts["symbol_box_correct"] += symbol_box_correct
                breakdown_counts["text_count_correct"] += text_count_correct
                breakdown_counts["symbol_class_correct"] += symbol_class_correct

                is_exact_entity = (
                    symbol_box_correct
                    and symbol_class_correct
                    and text_count_correct
                    and all_text_boxes_correct
                    and all_texts_correct
                )
                if bool(is_exact_entity):
                    image_true_positive += 1

        # 与截断GT匹配的预测从误检中移除，但截断GT本身也不计漏检。
        ignored_predictions = 0
        if num_predictions > 0 and num_targets > 0:
            ignored_predictions = sum(
                not bool(target_text_supervision[target_index])
                for prediction_index, target_index in zip(
                    prediction_indices,
                    target_indices,
                )
                if bool(valid_symbol_pairs[prediction_index, target_index])
            )
        image_false_positive = (
            num_predictions - ignored_predictions - image_true_positive
        )
        image_false_negative = complete_target_count - image_true_positive
        true_positive += image_true_positive
        false_positive += image_false_positive
        false_negative += image_false_negative
    metrics = summarize_entity_metrics(
        true_positive,
        false_positive,
        false_negative,
    )
    metrics["breakdown_counts"] = breakdown_counts
    return metrics


def entity_collate_fn(batch):
    """堆叠等尺寸图片，同时保留每张图片数量不同的实体标注。"""
    images, targets = zip(*batch)
    image_shapes = {tuple(image.shape) for image in images}
    if len(image_shapes) != 1:
        raise ValueError("同一batch中的图片尺寸不同，请设置统一的image_size")
    return torch.stack(images), list(targets)
