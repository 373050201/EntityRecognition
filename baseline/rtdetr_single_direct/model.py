"""
基于官方 RT-DETRv2-R18 的实体检测与文字识别模型。

RT-DETRv2 负责图元分类和图元框定位；同一个 query 继续预测文字框、锁状态和编号，
从而保持“一个 query 对应一个完整实体”的任务定义。
"""
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


class TextRecognitionHead(nn.Module):
    """从每个 query 对应的文字区域中提取序列特征并预测数字。"""

    def __init__(
        self,
        feature_channels,
        query_dim,
        hidden_dim=128,
        sequence_length=24,
        num_char_classes=CTC_NUM_CLASSES,
    ):
        super().__init__()
        if sequence_length <= 0:
            raise ValueError("sequence_length 必须大于0")

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
        self.character_head = nn.Linear(hidden_dim, num_char_classes)

    @staticmethod
    def _build_rois(text_boxes, feature_height, feature_width):
        """将归一化 cxcywh 文字框转换为 RoIAlign 使用的特征图坐标。"""
        batch_size, num_queries = text_boxes.shape[:2]
        boxes_xyxy = cxcywh_to_xyxy(text_boxes)

        minimum_width = 1e-3
        minimum_height = 1e-3
        x1 = boxes_xyxy[..., 0].clamp(0, 1 - minimum_width)
        y1 = boxes_xyxy[..., 1].clamp(0, 1 - minimum_height)
        x2 = boxes_xyxy[..., 2].clamp(minimum_width, 1)
        y2 = boxes_xyxy[..., 3].clamp(minimum_height, 1)
        x2 = torch.maximum(x2, x1 + minimum_width)
        y2 = torch.maximum(y2, y1 + minimum_height)

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
        """
        Args:
            ocr_features: `[B, 128, H8, W8]` 的高分辨率图像特征。
            query_features: `[B, Q, 256]` 的 RT-DETRv2 query 特征。
            text_boxes: `[B, Q, 4]` 的归一化 cxcywh 文字框。
        Returns:
            `[B, Q, 24, 11]` 的字符 logits，最后一类为 CTC blank。
        """
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
        character_logits = self.character_head(encoded_sequence)
        return character_logits.reshape(
            batch_size,
            num_queries,
            self.sequence_length,
            -1,
        )


class MLP(nn.Module):
    """用于边界框回归的三层感知机。"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(num_layers)
        )

    def forward(self, features):
        for index, layer in enumerate(self.layers):
            features = layer(features)
            if index < len(self.layers) - 1:
                features = torch.relu(features)
        return features


class SingleReferenceTextBoxHead(nn.Module):
    """仅使用图元query及图元框回归文字框，不读取独立文字区域特征。"""

    def __init__(self, hidden_dim, num_decoder_layers, mode):
        super().__init__()
        valid_modes = {"final_query", "iterative_symbol_query"}
        if mode not in valid_modes:
            raise ValueError(f"不支持的文字框回归模式：{mode}")

        self.mode = mode
        if mode == "final_query":
            self.final_head = MLP(hidden_dim, hidden_dim, 4)
        else:
            self.layer_heads = nn.ModuleList(
                MLP(hidden_dim, hidden_dim, 4)
                for _ in range(num_decoder_layers)
            )

    def forward(self, decoder_query_features, decoder_symbol_boxes):
        """
        Args:
            decoder_query_features: `[L, B, Q, C]`，各层图元query特征。
            decoder_symbol_boxes: `[L, B, Q, 4]`，各层归一化图元框。
        Returns:
            最终文字框`[B, Q, 4]`及前面各层的辅助文字框输出。
        """
        if self.mode == "final_query":
            # 单次直接回归基线：最终图元query一次性回归绝对文字框。
            final_boxes = self.final_head(decoder_query_features[-1]).sigmoid()
            return final_boxes, []

        if len(decoder_query_features) != len(self.layer_heads):
            raise ValueError("decoder层数与逐层文字框回归头数量不一致")

        layer_text_boxes = []
        for layer_index, (query_features, symbol_boxes) in enumerate(
            zip(decoder_query_features, decoder_symbol_boxes)
        ):
            # 逐层回归基线：使用当前层图元框，不在文字区域独立取样。
            symbol_logits = torch.logit(symbol_boxes.detach().clamp(1e-4, 1 - 1e-4))
            text_residual = self.layer_heads[layer_index](query_features)
            layer_text_boxes.append((symbol_logits + text_residual).sigmoid())

        return layer_text_boxes[-1], [
            {"pred_text_boxes": text_boxes}
            for text_boxes in layer_text_boxes[:-1]
        ]


class EntityRTDETR(nn.Module):
    """在官方 RT-DETRv2-R18 上增加实体文字框、锁状态和 OCR 分支。"""

    def __init__(
        self,
        num_classes=4,
        num_queries=100,
        pretrained_weights=None,
        sequence_length=24,
        text_box_mode="final_query",
    ):
        super().__init__()
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
        # 整体检测权重稍后统一加载，避免PResNet单独联网下载ImageNet权重。
        official.yaml_cfg["PResNet"]["pretrained"] = False
        official.yaml_cfg["RTDETRTransformerv2"]["num_queries"] = num_queries
        official.yaml_cfg["RTDETRTransformerv2"][
            "single_reference_text"
        ] = True
        detector = official.model

        if pretrained_weights:
            self._load_pretrained_detector(detector, pretrained_weights)

        self.backbone = detector.backbone
        self.encoder = detector.encoder
        self.decoder = detector.decoder
        self.num_classes = num_classes
        self.num_queries = num_queries
        hidden_dim = self.decoder.hidden_dim

        self.text_box_head = SingleReferenceTextBoxHead(
            hidden_dim=hidden_dim,
            num_decoder_layers=self.decoder.num_layers,
            mode=text_box_mode,
        )
        self.locked_head = nn.Linear(hidden_dim, 1)
        self.text_recognition_head = TextRecognitionHead(
            feature_channels=self.backbone.out_channels[0],
            query_dim=hidden_dim,
            hidden_dim=128,
            sequence_length=sequence_length,
        )

    @staticmethod
    def _load_pretrained_detector(detector, checkpoint_path):
        """只加载名称和 shape 都匹配的官方权重，跳过COCO分类头。"""
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

    @staticmethod
    def _prepare_detector_targets(targets, device):
        if targets is None:
            return None
        return [
            {
                "labels": target["labels"].to(device),
                "boxes": target["symbol_boxes"].to(device),
            }
            for target in targets
        ]

    def forward(self, images, targets=None, recognize_text=True):
        """
        Args:
            images: `[B, 3, H, W]`，像素范围为 `[0, 1]`。
            targets: 训练时用于 RT-DETRv2 去噪 query 的实体标签。
            recognize_text: 是否立刻使用预测文字框运行 OCR。
        """
        detector_targets = self._prepare_detector_targets(targets, images.device)
        backbone_features = self.backbone(images)
        # 三尺度特征分别约为[B,128,H/8,W/8]、[B,256,H/16,W/16]、[B,512,H/32,W/32]。
        encoded_features = self.encoder(backbone_features)
        outputs = self.decoder(encoded_features, detector_targets)

        query_features = outputs["query_features"]
        outputs["pred_symbol_boxes"] = outputs["pred_boxes"]
        pred_text_boxes, text_aux_outputs = self.text_box_head(
            outputs["decoder_query_features"],
            outputs["decoder_symbol_boxes"],
        )
        outputs["pred_text_boxes"] = pred_text_boxes
        outputs["text_aux_outputs"] = text_aux_outputs
        outputs["pred_locked"] = self.locked_head(query_features)
        outputs["ocr_features"] = backbone_features[0]

        if recognize_text:
            outputs["pred_identifier_logits"] = self.text_recognition_head(
                outputs["ocr_features"],
                query_features,
                pred_text_boxes,
            )
        return outputs


def build_model(
    pretrained_weights=None,
    num_classes=4,
    num_queries=100,
    sequence_length=24,
    text_box_mode="final_query",
):
    """构建使用官方 RT-DETRv2-R18 检测核心的实体模型。"""
    return EntityRTDETR(
        num_classes=num_classes,
        num_queries=num_queries,
        pretrained_weights=pretrained_weights,
        sequence_length=sequence_length,
        text_box_mode=text_box_mode,
    )
