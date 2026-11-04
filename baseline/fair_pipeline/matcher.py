"""使用纯几何代价和匈牙利算法关联独立图元与文字预测。"""
import math
import torch
from scipy.optimize import linear_sum_assignment
from utils import ctc_greedy_decode, cxcywh_to_xyxy



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


@torch.no_grad()
def associate_predictions(
    symbol_outputs,
    text_outputs,
    symbol_conf_threshold=0.5,
    text_conf_threshold=0.5,
    lock_threshold=0.5,
    max_distance=0.06,
):
    """过滤两类预测，并用匈牙利算法完成每张图的一对一几何匹配。"""
    decoded_identifiers = ctc_greedy_decode(
        text_outputs["pred_identifier_logits"]
    )
    batch_predictions = []
    batch_size = symbol_outputs["pred_logits"].shape[0]
    for batch_index in range(batch_size):
        symbol_scores, symbol_labels = symbol_outputs["pred_logits"][
            batch_index
        ].sigmoid().max(-1)
        symbol_keep = symbol_scores >= symbol_conf_threshold
        kept_symbol_scores = symbol_scores[symbol_keep]
        kept_symbol_labels = symbol_labels[symbol_keep]
        kept_symbol_locked = (
            symbol_outputs["pred_locked"][batch_index, symbol_keep]
            .sigmoid()
            .squeeze(-1)
            >= lock_threshold
        )
        kept_symbol_boxes_cxcywh = symbol_outputs["pred_boxes"][
            batch_index,
            symbol_keep,
        ].clamp(0, 1)
        kept_symbol_boxes = cxcywh_to_xyxy(
            kept_symbol_boxes_cxcywh
        ).clamp(0, 1)

        text_scores = text_outputs["pred_logits"][
            batch_index
        ].sigmoid().squeeze(-1)
        text_keep = text_scores >= text_conf_threshold
        text_query_indices = torch.arange(
            len(text_scores),
            device=text_scores.device,
        )[text_keep]
        kept_text_scores = text_scores[text_keep]
        kept_text_boxes = cxcywh_to_xyxy(
            text_outputs["pred_boxes"][batch_index, text_keep]
        ).clamp(0, 1)
        kept_identifiers = [
            decoded_identifiers[batch_index][query_index]
            for query_index in text_query_indices.tolist()
        ]

        empty_prediction = {
            "scores": kept_symbol_scores.new_empty((0,)),
            "symbol_boxes": kept_symbol_boxes.new_empty((0, 4)),
            "symbol_labels": kept_symbol_labels.new_empty((0,), dtype=torch.long),
            "locked": kept_symbol_locked.new_empty((0,), dtype=torch.bool),
            "text_boxes": kept_text_boxes.new_empty((0, 4)),
            "identifiers": [],
            "association_distances": kept_symbol_scores.new_empty((0,)),
        }
        if len(kept_symbol_boxes) == 0 or len(kept_text_boxes) == 0:
            batch_predictions.append(empty_prediction)
            continue

        distances = _point_to_box_distance(
            kept_symbol_boxes_cxcywh[:, :2],
            kept_text_boxes,
        )
        symbol_indices, text_indices = linear_sum_assignment(
            distances.detach().cpu().numpy()
        )
        valid_pairs = [
            (symbol_index, text_index)
            for symbol_index, text_index in zip(symbol_indices, text_indices)
            if distances[symbol_index, text_index] <= max_distance
        ]
        if not valid_pairs:
            batch_predictions.append(empty_prediction)
            continue

        matched_symbol_indices = torch.tensor(
            [pair[0] for pair in valid_pairs],
            dtype=torch.long,
            device=kept_symbol_boxes.device,
        )
        matched_text_indices = torch.tensor(
            [pair[1] for pair in valid_pairs],
            dtype=torch.long,
            device=kept_text_boxes.device,
        )
        matched_scores = torch.minimum(
            kept_symbol_scores[matched_symbol_indices],
            kept_text_scores[matched_text_indices],
        )
        batch_predictions.append(
            {
                "scores": matched_scores,
                "symbol_boxes": kept_symbol_boxes[matched_symbol_indices],
                "symbol_labels": kept_symbol_labels[matched_symbol_indices],
                "locked": kept_symbol_locked[matched_symbol_indices],
                "text_boxes": kept_text_boxes[matched_text_indices],
                "identifiers": [
                    kept_identifiers[index]
                    for index in matched_text_indices.tolist()
                ],
                "association_distances": distances[
                    matched_symbol_indices,
                    matched_text_indices,
                ],
            }
        )
    return batch_predictions
