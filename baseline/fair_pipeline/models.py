"""公平流水线中相互独立的图元RT-DETRv2与文字RT-DETRv2模型。"""
import sys
from pathlib import Path
import torch
from torch import nn
from torchvision.ops import roi_align
from utils import CTC_NUM_CLASSES, cxcywh_to_xyxy



PROJECT_ROOT = Path(__file__).resolve().parents[2]
RTDETR_ROOT = PROJECT_ROOT / "third_party" / "rtdetrv2"
if str(RTDETR_ROOT) not in sys.path:
    sys.path.insert(0, str(RTDETR_ROOT))
from src.core import YAMLConfig  # noqa: E402


def _build_detector(num_classes, num_queries):
    """构建当前项目使用的RT-DETRv2-R18检测核心。"""
    official_config = (
        RTDETR_ROOT
        / "configs"
        / "rtdetrv2"
        / "rtdetrv2_r18vd_120e_coco.yml"
    )
    official = YAMLConfig(
        str(official_config),
        num_classes=num_classes,
        eval_spatial_size=None,
    )
    official.yaml_cfg["PResNet"]["pretrained"] = False
    transformer_config = official.yaml_cfg["RTDETRTransformerv2"]
    transformer_config["num_queries"] = num_queries
    # 两个基线仅使用主检测流，关闭当前工程decoder中的跨区域交互与训练扰动。
    transformer_config["text_reference_jitter_scale"] = 0.0
    transformer_config["region_interaction_mode"] = "none"
    transformer_config["primary_only"] = True
    detector = official.model
    # primary_only前向不读取这些扩展模块，移除它们以避免无用参数进入基线权重。
    detector.decoder.text_ref_point_head = None
    detector.decoder.dec_text_bbox_head = None
    for layer in detector.decoder.decoder.layers:
        layer.text_cross_attn = None
        layer.text_dropout = None
        layer.text_norm = None
    return detector


def _load_pretrained_detector(detector, checkpoint_path):
    """加载名称和shape兼容的COCO预训练权重。"""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"RT-DETRv2预训练权重不存在：{checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    if isinstance(state_dict, dict) and "module" in state_dict:
        state_dict = state_dict["module"]
    current_state = detector.state_dict()
    compatible_state = {
        name: value
        for name, value in state_dict.items()
        if name in current_state and current_state[name].shape == value.shape
    }
    detector.load_state_dict(compatible_state, strict=False)
    print(
        "RT-DETRv2预训练权重："
        f"成功加载{len(compatible_state)}/{len(current_state)}个参数张量"
    )


class TextRecognitionHead(nn.Module):
    """仅使用文字检测器自身的query和文字区域特征识别编号。"""

    def __init__(
        self,
        feature_channels,
        query_dim,
        hidden_dim=128,
        sequence_length=24,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_projection = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.sequence_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.character_head = nn.Linear(hidden_dim, CTC_NUM_CLASSES)

    @staticmethod
    def _build_rois(text_boxes, feature_height, feature_width):
        """把归一化文字框转换为RoIAlign需要的特征图坐标。"""
        batch_size, num_queries = text_boxes.shape[:2]
        boxes_xyxy = cxcywh_to_xyxy(text_boxes)
        minimum_size = 1e-3
        x1 = boxes_xyxy[..., 0].clamp(0, 1 - minimum_size)
        y1 = boxes_xyxy[..., 1].clamp(0, 1 - minimum_size)
        x2 = boxes_xyxy[..., 2].clamp(minimum_size, 1)
        y2 = boxes_xyxy[..., 3].clamp(minimum_size, 1)
        x2 = torch.maximum(x2, x1 + minimum_size)
        y2 = torch.maximum(y2, y1 + minimum_size)
        feature_boxes = torch.stack(
            (
                x1 * feature_width,
                y1 * feature_height,
                x2 * feature_width,
                y2 * feature_height,
            ),
            dim=-1,
        ).reshape(batch_size * num_queries, 4)
        batch_indices = torch.arange(
            batch_size,
            dtype=feature_boxes.dtype,
            device=feature_boxes.device,
        ).repeat_interleave(num_queries)
        return torch.cat((batch_indices[:, None], feature_boxes), dim=1)

    def forward(self, ocr_features, query_features, text_boxes):
        batch_size, num_queries = query_features.shape[:2]
        feature_height, feature_width = ocr_features.shape[-2:]
        projected_features = self.feature_projection(ocr_features)
        rois = self._build_rois(text_boxes, feature_height, feature_width)
        region_features = roi_align(
            projected_features,
            rois,
            output_size=(4, self.sequence_length),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        region_sequence = region_features.mean(dim=2).transpose(1, 2)
        query_sequence = self.query_projection(
            query_features.reshape(batch_size * num_queries, -1)
        )[:, None, :]
        fused_sequence = self.fusion_norm(region_sequence + query_sequence)
        encoded_sequence, _ = self.sequence_encoder(fused_sequence)
        logits = self.character_head(encoded_sequence)
        return logits.reshape(
            batch_size,
            num_queries,
            self.sequence_length,
            -1,
        )


class SymbolDetector(nn.Module):
    """独立预测图元框、图元类别和锁状态的RT-DETRv2。"""

    def __init__(
        self,
        num_classes=4,
        num_queries=100,
        pretrained_weights=None,
    ):
        super().__init__()
        detector = _build_detector(num_classes, num_queries)
        if pretrained_weights:
            _load_pretrained_detector(detector, pretrained_weights)
        self.backbone = detector.backbone
        self.encoder = detector.encoder
        self.decoder = detector.decoder
        self.locked_head = nn.Linear(self.decoder.hidden_dim, 1)

    @staticmethod
    def _prepare_targets(targets, device):
        if targets is None:
            return None
        return [
            {
                "labels": target["symbol_labels"].to(device),
                "boxes": target["symbol_boxes"].to(device),
            }
            for target in targets
        ]

    def forward(self, images, targets=None):
        detector_targets = self._prepare_targets(targets, images.device)
        features = self.backbone(images)
        encoded_features = self.encoder(features)
        outputs = self.decoder(encoded_features, detector_targets)
        outputs["pred_locked"] = self.locked_head(outputs["query_features"])
        return outputs


class TextSpotter(nn.Module):
    """独立预测文字框并识别编号的RT-DETRv2文字检测识别器。"""

    def __init__(
        self,
        num_queries=100,
        sequence_length=24,
        pretrained_weights=None,
    ):
        super().__init__()
        detector = _build_detector(1, num_queries)
        if pretrained_weights:
            _load_pretrained_detector(detector, pretrained_weights)
        self.backbone = detector.backbone
        self.encoder = detector.encoder
        self.decoder = detector.decoder
        self.text_recognition_head = TextRecognitionHead(
            feature_channels=self.backbone.out_channels[0],
            query_dim=self.decoder.hidden_dim,
            sequence_length=sequence_length,
        )

    @staticmethod
    def _prepare_targets(targets, device):
        if targets is None:
            return None
        return [
            {
                "labels": target["text_labels"].to(device),
                "boxes": target["text_boxes"].to(device),
            }
            for target in targets
        ]

    def forward(self, images, targets=None, recognize_text=True):
        detector_targets = self._prepare_targets(targets, images.device)
        features = self.backbone(images)
        encoded_features = self.encoder(features)
        outputs = self.decoder(encoded_features, detector_targets)
        outputs["ocr_features"] = features[0]
        if recognize_text:
            outputs["pred_identifier_logits"] = self.text_recognition_head(
                outputs["ocr_features"],
                outputs["query_features"],
                outputs["pred_boxes"],
            )
        return outputs


def build_symbol_detector(
    pretrained_weights=None,
    num_classes=4,
    num_queries=100,
):
    """构建公平流水线的图元检测器。"""
    return SymbolDetector(
        num_classes=num_classes,
        num_queries=num_queries,
        pretrained_weights=pretrained_weights,
    )


def build_text_spotter(
    pretrained_weights=None,
    num_queries=100,
    sequence_length=24,
):
    """构建公平流水线的文字检测识别器。"""
    return TextSpotter(
        num_queries=num_queries,
        sequence_length=sequence_length,
        pretrained_weights=pretrained_weights,
    )
