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
PROJECT_ROOT = Path(__file__).resolve().parent
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
        guidance_mode="none",
        guidance_num_heads=4,
        guidance_gate_mode="none",
    ):
        super().__init__()
        if sequence_length <= 0:
            raise ValueError("sequence_length 必须大于0")
        if guidance_mode not in {"none", "cross_attention"}:
            raise ValueError(f"不支持的OCR指导模式：{guidance_mode}")
        if hidden_dim % guidance_num_heads != 0:
            raise ValueError("OCR隐藏维度必须能被指导注意力头数整除")
        if guidance_gate_mode not in {"none", "dynamic", "text_only"}:
            raise ValueError(f"不支持的OCR指导门控模式：{guidance_gate_mode}")
        if guidance_gate_mode != "none" and guidance_mode != "cross_attention":
            raise ValueError("OCR指导来源控制只能用于cross_attention模式")

        self.sequence_length = sequence_length
        self.guidance_mode = guidance_mode
        self.guidance_gate_mode = guidance_gate_mode
        self.feature_projection = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.query_projection = nn.Linear(query_dim, hidden_dim)
        self.text_query_projection = nn.Linear(query_dim, hidden_dim)
        self.guidance_attention = nn.MultiheadAttention(
            hidden_dim,
            guidance_num_heads,
            batch_first=True,
        )
        self.guidance_dropout = nn.Dropout(0.0)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.sequence_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.character_head = nn.Linear(hidden_dim, num_char_classes)

        # 放在V2-A全部模块之后构造，避免新增模块改变共享参数的随机初始化顺序。
        self.guidance_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # V2-B从g=0.5开始；配合2倍等幅补偿，初始引导记忆与V2-A完全相同。
        nn.init.zeros_(self.guidance_gate[2].weight)
        nn.init.zeros_(self.guidance_gate[2].bias)

    def _build_guidance_memory(self, symbol_token, text_token):
        """按实体构造图元、编号双token引导记忆。"""
        if self.guidance_gate_mode == "none":
            return torch.cat([symbol_token, text_token], dim=1)
        if self.guidance_gate_mode == "text_only":
            # V2-C固定在V2-B的g=0端点，保留双token长度和2倍等幅约定。
            return torch.cat(
                [torch.zeros_like(symbol_token), 2.0 * text_token],
                dim=1,
            )

        gate_input = torch.cat(
            [symbol_token.squeeze(1), text_token.squeeze(1)],
            dim=-1,
        )
        symbol_ratio = self.guidance_gate(gate_input)[:, None, :]
        symbol_token = 2.0 * symbol_ratio * symbol_token
        text_token = 2.0 * (1.0 - symbol_ratio) * text_token
        return torch.cat([symbol_token, text_token], dim=1)

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

    def forward(
        self,
        ocr_features,
        query_features,
        text_query_features,
        text_boxes,
    ):
        """
        Args:
            ocr_features: `[B, 128, H8, W8]` 的高分辨率图像特征。
            query_features: `[B, Q, 256]` 的 RT-DETRv2 query 特征。
            text_query_features: `[B, Q, 256]` 的文字区域私有特征。
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
        symbol_token = self.query_projection(
            query_features.reshape(batch_size * num_queries, -1)
        )[:, None, :]
        if self.guidance_mode == "cross_attention":
            text_token = self.text_query_projection(
                text_query_features.reshape(batch_size * num_queries, -1)
            )[:, None, :]
            # V2-A固定双token；V2-B动态门控；V2-C固定只使用文字token。
            guidance_memory = self._build_guidance_memory(
                symbol_token,
                text_token,
            )
            guidance, _ = self.guidance_attention(
                region_sequence,
                guidance_memory,
                guidance_memory,
                need_weights=False,
            )
            fused_sequence = self.fusion_norm(
                region_sequence + self.guidance_dropout(guidance)
            )
        else:
            # V1-B对照：保持原始图元query广播相加的数据流。
            fused_sequence = self.fusion_norm(
                region_sequence + symbol_token
            )
        encoded_sequence, _ = self.sequence_encoder(fused_sequence)
        character_logits = self.character_head(encoded_sequence)
        return character_logits.reshape(
            batch_size,
            num_queries,
            self.sequence_length,
            -1,
        )


class EntityRTDETR(nn.Module):
    """在官方 RT-DETRv2-R18 上增加实体文字框、锁状态和 OCR 分支。"""

    def __init__(
        self,
        num_classes=4,
        num_queries=100,
        pretrained_weights=None,
        sequence_length=24,
        text_reference_offset_scale=5.0,
        text_reference_jitter_scale=0.2,
        text_first_layer_sampling_scale=5.0,
        text_refine_sampling_scale=0.5,
        region_interaction_mode="none",
        region_interaction_init_scale=0.001,
        ocr_guidance_mode="none",
        ocr_guidance_num_heads=4,
        ocr_guidance_gate_mode="none",
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
            "text_reference_offset_scale"
        ] = text_reference_offset_scale
        official.yaml_cfg["RTDETRTransformerv2"][
            "text_reference_jitter_scale"
        ] = text_reference_jitter_scale
        official.yaml_cfg["RTDETRTransformerv2"][
            "text_first_layer_sampling_scale"
        ] = text_first_layer_sampling_scale
        official.yaml_cfg["RTDETRTransformerv2"][
            "text_refine_sampling_scale"
        ] = text_refine_sampling_scale
        official.yaml_cfg["RTDETRTransformerv2"][
            "region_interaction_mode"
        ] = region_interaction_mode
        official.yaml_cfg["RTDETRTransformerv2"][
            "region_interaction_init_scale"
        ] = region_interaction_init_scale
        detector = official.model

        if pretrained_weights:
            self._load_pretrained_detector(detector, pretrained_weights)
        # 新增文字采样分支复用已加载的图元注意力作为稳定初始化。
        detector.decoder.initialize_text_attention_from_symbol()

        self.backbone = detector.backbone
        self.encoder = detector.encoder
        self.decoder = detector.decoder
        self.num_classes = num_classes
        self.num_queries = num_queries
        hidden_dim = self.decoder.hidden_dim

        self.locked_head = nn.Linear(hidden_dim, 1)
        self.text_recognition_head = TextRecognitionHead(
            feature_channels=self.backbone.out_channels[0],
            query_dim=hidden_dim,
            hidden_dim=128,
            sequence_length=sequence_length,
            guidance_mode=ocr_guidance_mode,
            guidance_num_heads=ocr_guidance_num_heads,
            guidance_gate_mode=ocr_guidance_gate_mode,
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
        # `[B, Q, 4]`：文字框已在RT-DETRv2 decoder内部完成区域采样和逐层回归。
        pred_text_boxes = outputs["pred_text_boxes"]
        outputs["pred_locked"] = self.locked_head(query_features)
        outputs["ocr_features"] = backbone_features[0]

        if recognize_text:
            outputs["pred_identifier_logits"] = self.text_recognition_head(
                outputs["ocr_features"],
                query_features,
                outputs["text_query_features"],
                pred_text_boxes,
            )
        return outputs


def build_model(
    pretrained_weights=None,
    num_classes=4,
    num_queries=100,
    sequence_length=24,
    text_reference_offset_scale=5.0,
    text_reference_jitter_scale=0.2,
    text_first_layer_sampling_scale=5.0,
    text_refine_sampling_scale=0.5,
    region_interaction_mode="none",
    region_interaction_init_scale=0.001,
    ocr_guidance_mode="none",
    ocr_guidance_num_heads=4,
    ocr_guidance_gate_mode="none",
):
    """构建使用官方 RT-DETRv2-R18 检测核心的实体模型。"""
    return EntityRTDETR(
        num_classes=num_classes,
        num_queries=num_queries,
        pretrained_weights=pretrained_weights,
        sequence_length=sequence_length,
        text_reference_offset_scale=text_reference_offset_scale,
        text_reference_jitter_scale=text_reference_jitter_scale,
        text_first_layer_sampling_scale=text_first_layer_sampling_scale,
        text_refine_sampling_scale=text_refine_sampling_scale,
        region_interaction_mode=region_interaction_mode,
        region_interaction_init_scale=region_interaction_init_scale,
        ocr_guidance_mode=ocr_guidance_mode,
        ocr_guidance_num_heads=ocr_guidance_num_heads,
        ocr_guidance_gate_mode=ocr_guidance_gate_mode,
    )
