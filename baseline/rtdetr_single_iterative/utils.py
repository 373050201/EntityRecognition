"""
项目通用工具函数
"""
import random
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou



CTC_CHARACTERS = "0123456789"
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


def encode_identifiers(identifiers, device=None):
    """将数字字符串列表编码为 CTCLoss 使用的一维目标和长度。"""
    encoded = []
    lengths = []
    for identifier in identifiers:
        identifier = str(identifier)
        if not identifier or any(character not in CTC_CHARACTERS for character in identifier):
            raise ValueError(f"编号必须是非空纯数字字符串，实际为：{identifier!r}")
        encoded.extend(int(character) for character in identifier)
        lengths.append(len(identifier))

    return (
        torch.tensor(encoded, dtype=torch.long, device=device),
        torch.tensor(lengths, dtype=torch.long, device=device),
    )


def ctc_greedy_decode(identifier_logits, blank_index=CTC_BLANK_INDEX):
    """
    将 `[B, Q, T, C]` 的字符 logits 贪心解码为二维编号字符串列表。

    解码顺序是先合并相邻重复类别，再删除 CTC blank。
    """
    if identifier_logits.ndim != 4:
        raise ValueError("identifier_logits 的 shape 必须为 [B, Q, T, C]")
    if not 0 <= blank_index < identifier_logits.shape[-1]:
        raise ValueError("blank_index 超出字符类别范围")

    predicted_indices = identifier_logits.argmax(dim=-1).detach().cpu()
    decoded_batch = []
    for image_predictions in predicted_indices:
        decoded_queries = []
        for sequence in image_predictions:
            previous_index = None
            characters = []
            for character_index in sequence.tolist():
                if character_index != previous_index and character_index != blank_index:
                    if character_index >= len(CTC_CHARACTERS):
                        raise ValueError("预测字符索引超出当前数字字符集")
                    characters.append(CTC_CHARACTERS[character_index])
                previous_index = character_index
            decoded_queries.append("".join(characters))
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
        "num_characters": 0,
        "symbol_iou_sum": 0.0,
        "text_iou_sum": 0.0,
        "symbol_box_correct": 0,
        "text_box_correct": 0,
        "symbol_class_correct": 0,
        "locked_correct": 0,
        "identifier_character_correct": 0,
        "identifier_exact_correct": 0,
    }


def summarize_entity_breakdown(counts):
    """将验证集累计的子任务计数转换为便于诊断的比例指标。"""
    num_targets = counts["num_targets"]
    num_characters = counts["num_characters"]

    def entity_rate(name):
        return counts[name] / num_targets if num_targets > 0 else 0.0

    return {
        "symbol_mean_iou": entity_rate("symbol_iou_sum"),
        "symbol_box_accuracy": entity_rate("symbol_box_correct"),
        "text_mean_iou": entity_rate("text_iou_sum"),
        "text_box_accuracy": entity_rate("text_box_correct"),
        "symbol_class_accuracy": entity_rate("symbol_class_correct"),
        "locked_accuracy": entity_rate("locked_correct"),
        "identifier_character_accuracy": (
            counts["identifier_character_correct"] / num_characters
            if num_characters > 0 else 0.0
        ),
        "identifier_exact_accuracy": entity_rate("identifier_exact_correct"),
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
    lock_threshold=0.5,
):
    """对使用 sigmoid 分类的 RT-DETRv2 输出执行实体级严格评估。"""
    thresholds = (
        entity_conf_threshold,
        symbol_iou_threshold,
        text_iou_threshold,
        lock_threshold,
    )
    if any(threshold < 0 or threshold > 1 for threshold in thresholds):
        raise ValueError("实体评估的所有阈值都必须位于[0, 1]")

    batch_size, num_queries = outputs["pred_logits"].shape[:2]
    if len(targets) != batch_size:
        raise ValueError("targets数量必须与batch size一致")

    decoded_identifiers = ctc_greedy_decode(outputs["pred_identifier_logits"])
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
        predicted_locked = (
            outputs["pred_locked"][batch_index, keep].sigmoid().squeeze(-1)
            >= lock_threshold
        )
        predicted_symbol_boxes = cxcywh_to_xyxy(
            outputs["pred_symbol_boxes"][batch_index, keep]
        ).clamp(0, 1)
        predicted_text_boxes = cxcywh_to_xyxy(
            outputs["pred_text_boxes"][batch_index, keep]
        ).clamp(0, 1)
        predicted_identifiers = [
            decoded_identifiers[batch_index][query_index]
            for query_index in kept_query_indices.tolist()
        ]

        target_labels = targets[batch_index]["labels"].to(predicted_labels.device)
        target_locked = targets[batch_index]["locked"].to(predicted_labels.device)
        target_symbol_boxes = cxcywh_to_xyxy(
            targets[batch_index]["symbol_boxes"].to(predicted_labels.device)
        ).clamp(0, 1)
        target_text_boxes = cxcywh_to_xyxy(
            targets[batch_index]["text_boxes"].to(predicted_labels.device)
        ).clamp(0, 1)
        target_identifiers = targets[batch_index]["identifiers"]

        num_predictions = len(predicted_labels)
        num_targets = len(target_labels)
        image_true_positive = 0
        target_identifier_strings = [
            str(identifier) for identifier in target_identifiers
        ]
        breakdown_counts["num_targets"] += num_targets
        breakdown_counts["num_characters"] += sum(
            len(identifier) for identifier in target_identifier_strings
        )

        if num_predictions > 0 and num_targets > 0:
            symbol_ious = box_iou(predicted_symbol_boxes, target_symbol_boxes)
            text_ious = box_iou(predicted_text_boxes, target_text_boxes)
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
                text_iou = text_ious[prediction_index, target_index].item()
                symbol_box_correct = bool(
                    valid_symbol_pairs[prediction_index, target_index]
                )
                text_box_correct = text_iou >= text_iou_threshold
                symbol_class_correct = bool(
                    predicted_labels[prediction_index]
                    == target_labels[target_index]
                )
                locked_correct = bool(
                    predicted_locked[prediction_index]
                    == target_locked[target_index].bool()
                )
                predicted_identifier = predicted_identifiers[prediction_index]
                target_identifier = target_identifier_strings[target_index]
                identifier_exact_correct = (
                    predicted_identifier == target_identifier
                )

                breakdown_counts["symbol_iou_sum"] += symbol_iou
                breakdown_counts["text_iou_sum"] += text_iou
                breakdown_counts["symbol_box_correct"] += symbol_box_correct
                breakdown_counts["text_box_correct"] += text_box_correct
                breakdown_counts["symbol_class_correct"] += symbol_class_correct
                breakdown_counts["locked_correct"] += locked_correct
                breakdown_counts["identifier_exact_correct"] += (
                    identifier_exact_correct
                )
                identifier_distance = _levenshtein_distance(
                    predicted_identifier,
                    target_identifier,
                )
                breakdown_counts["identifier_character_correct"] += max(
                    len(target_identifier) - identifier_distance,
                    0,
                )

                is_exact_entity = (
                    symbol_box_correct
                    and symbol_class_correct
                    and locked_correct
                    and text_box_correct
                    and identifier_exact_correct
                )
                if bool(is_exact_entity):
                    image_true_positive += 1

        image_false_positive = num_predictions - image_true_positive
        image_false_negative = num_targets - image_true_positive
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
