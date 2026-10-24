"""
基于实体查询的简易 DETR 模型。

一个 query 对应一个完整实体，并同时预测图元类别、图元框、文字框、锁状态和编号。
文字识别分支根据每个 query 的文字框，从高分辨率特征中提取文字序列并供 CTC 训练。
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops import roi_align
from .utils import CTC_NUM_CLASSES, cxcywh_to_xyxy



class MLP(nn.Module):
    """用于边界框回归的多层感知机。"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for layer_index in range(num_layers):
            in_dim = input_dim if layer_index == 0 else hidden_dim
            out_dim = output_dim if layer_index == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_index < num_layers - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PositionEmbeddingSine(nn.Module):
    """为二维特征图生成 DETR 使用的正弦位置编码。"""

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True):
        super().__init__()
        if num_pos_feats % 2 != 0:
            raise ValueError("num_pos_feats 必须为偶数")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, feature_map, mask=None):
        # feature_map: [B, C, Hf, Wf]；mask: [B, Hf, Wf]
        batch_size, _, height, width = feature_map.shape
        if mask is None:
            mask = torch.zeros(
                (batch_size, height, width),
                dtype=torch.bool,
                device=feature_map.device,
            )

        not_mask = (~mask).float()
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=feature_map.device,
        )
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)

        # 横向和纵向位置编码各占一半通道。
        # pos: [B, Hf, Wf, C] -> [B, C, Hf, Wf]
        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos.permute(0, 3, 1, 2)


class ResNet18Backbone(nn.Module):
    """提取检测特征，并保留一层高分辨率特征供后续 CTC 使用。"""

    detection_channels = 256
    ocr_channels = 128

    def __init__(self, pretrained=True, train_backbone=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        if not train_backbone:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, images):
        # images: [B, 3, H, W]；以 540×960 为例。
        features = self.stem(images)
        # [B, 64, ceil(H/4), ceil(W/4)]，示例：[B, 64, 135, 240]
        features = self.layer1(features)
        # [B, 64, ceil(H/4), ceil(W/4)]，示例：[B, 64, 135, 240]
        ocr_features = self.layer2(features)
        # [B, 128, ceil(H/8), ceil(W/8)]，示例：[B, 128, 68, 120]
        detection_features = self.layer3(ocr_features)
        # [B, 256, ceil(H/16), ceil(W/16)]，示例：[B, 256, 34, 60]
        return detection_features, ocr_features


class TextRecognitionHead(nn.Module):
    """从每个 query 对应的文字区域中提取序列特征并预测数字。"""

    def __init__(
        self,
        feature_channels=128,
        query_dim=128,
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
        # text_boxes: [B, Q, 4]，最后一维为归一化[cx, cy, w, h]。
        boxes_xyxy = cxcywh_to_xyxy(text_boxes)
        # boxes_xyxy: [B, Q, 4]，最后一维为归一化[x1, y1, x2, y2]。

        # 预测框可能越过图像边界，裁剪后还需保证至少覆盖一个很小的区域。
        minimum_width = 1e-3
        minimum_height = 1e-3
        x1 = boxes_xyxy[..., 0].clamp(0, 1 - minimum_width)
        y1 = boxes_xyxy[..., 1].clamp(0, 1 - minimum_height)
        x2 = boxes_xyxy[..., 2].clamp(minimum_width, 1)
        y2 = boxes_xyxy[..., 3].clamp(minimum_height, 1)
        x2 = torch.maximum(x2, x1 + minimum_width)
        y2 = torch.maximum(y2, y1 + minimum_height)
        # x1/y1/x2/y2: [B, Q]，范围限制在[0, 1]附近且宽高不为0。

        feature_boxes = torch.stack(
            (
                x1 * feature_width,
                y1 * feature_height,
                x2 * feature_width,
                y2 * feature_height,
            ),
            dim=-1,
        ).reshape(batch_size * num_queries, 4)
        # feature_boxes: [B*Q, 4]，已换算为OCR特征图上的绝对坐标。
        batch_indices = torch.arange(
            batch_size,
            dtype=feature_boxes.dtype,
            device=feature_boxes.device,
        ).repeat_interleave(num_queries)
        # batch_indices: [B*Q]，说明每个文字框来自batch中的哪张图片。
        rois = torch.cat((batch_indices[:, None], feature_boxes), dim=1)
        # rois: [B*Q, 5]，每行为[batch_index, x1, y1, x2, y2]。
        return rois

    def forward(self, ocr_features, query_features, text_boxes):
        """
        Args:
            ocr_features: `[B, 128, H8, W8]` 的高分辨率图像特征。
            query_features: `[B, Q, 128]` 的实体 query 特征。
            text_boxes: `[B, Q, 4]` 的归一化 cxcywh 文字框。

        Returns:
            `[B, Q, T, 11]` 的字符 logits，最后一类是 CTC blank。
        """
        batch_size, num_queries = query_features.shape[:2]
        feature_height, feature_width = ocr_features.shape[-2:]
        # batch_size=B，num_queries=Q；feature_height=H8，feature_width=W8。
        projected_features = self.feature_projection(ocr_features)
        # [B, feature_channels, H8, W8] -> [B, hidden_dim, H8, W8]
        # 当前默认值下为[B, 128, H8, W8]，用于统一OCR分支的通道数。

        rois = self._build_rois(
            text_boxes,
            feature_height,
            feature_width,
        )
        # rois: [B*Q, 5]，每个query对应一个某张图上待裁剪的文字区域。
        region_features = roi_align(
            projected_features,
            rois,
            output_size=(4, self.sequence_length),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )
        # region_features: [B*Q, hidden_dim, 4, T]
        # 当前T=24，每个大小不同的OCR特征图裁剪区域都被对齐为4×24的固定尺寸。
        region_sequence = region_features.mean(dim=2).transpose(1, 2)
        # 沿列池化，即取mean后：[B*Q, hidden_dim, T]；transpose后：[B*Q, T, hidden_dim]
        # 当前默认值下为[B*Q, 24, 128]，水平方向24格作为CTC时间步。

        # 将同一个实体 query 的语义特征加入该文字区域的每个时间步。
        query_sequence = self.query_projection(
            query_features.reshape(batch_size * num_queries, -1)
        )[:, None, :]
        # reshape后：[B*Q, query_dim]；Linear后：[B*Q, hidden_dim]
        # 增加时间维后query_sequence：[B*Q, 1, hidden_dim]。
        fused_sequence = self.fusion_norm(region_sequence + query_sequence)
        # 广播相加时query会复制到T个位置，fused_sequence：[B*Q, T, hidden_dim]，即融合了query语义的文本序列特征
        encoded_sequence, _ = self.sequence_encoder(fused_sequence)
        # 双向LSTM两侧各输出hidden_dim/2，拼接后仍为[B*Q, T, hidden_dim]，进一步让时间步看到相邻信息
        character_logits = self.character_head(encoded_sequence)
        # character_logits: [B*Q, T, 11]，10个数字类别加1个CTC blank。
        pred_identifier_logits = character_logits.reshape(
            batch_size,
            num_queries,
            self.sequence_length,
            -1,
        )
        # pred_identifier_logits: [B, Q, T, 11]，当前为[B, 20, 24, 11]，为每个时间步预测一个数字或blank，符合CTC_loss输入要求
        return pred_identifier_logits


class EntityDETR(nn.Module):
    """使用固定数量实体 query 的简易 DETR。"""

    def __init__(
        self,
        num_classes=4,
        num_queries=20,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        pretrained_backbone=True,
        train_backbone=True,
    ):
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model 必须能被4整除，以构造二维位置编码")

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.d_model = d_model

        self.backbone = ResNet18Backbone(
            pretrained=pretrained_backbone,
            train_backbone=train_backbone,
        )
        self.input_projection = nn.Conv2d(
            self.backbone.detection_channels,
            d_model,
            kernel_size=1,
        )
        self.position_embedding = PositionEmbeddingSine(d_model // 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # Encoder 每层：[B, N, 128] -> FFN [B, N, 512] -> [B, N, 128]。
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # Decoder 每层让 [B, 20, 128] 的 query 读取 [B, N, 128] 的图像 memory。
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # 每个可学习 query 代表一个潜在的完整实体。
        self.query_embed = nn.Embedding(num_queries, d_model)

        # 四种图元类别之外，再增加一个 no-object 类。
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.symbol_bbox_head = MLP(d_model, d_model, 4, 3)
        self.text_bbox_head = MLP(d_model, d_model, 4, 3)
        self.locked_head = nn.Linear(d_model, 1)
        self.text_recognition_head = TextRecognitionHead(
            feature_channels=self.backbone.ocr_channels,
            query_dim=d_model,
            hidden_dim=d_model,
            sequence_length=24,
        )

    def forward(self, images, image_mask=None, recognize_text=True):
        """
        Args:
            images: `[B, 3, H, W]`，已经归一化的输入图像。
            image_mask: 可选的 `[B, H, W]` 布尔掩码，True 表示填充区域。
            recognize_text: 是否立即使用预测文字框执行OCR。训练时可先设为
                False，在Hungarian匹配后改用对齐的真实文字框执行OCR。

        Returns:
            分类、双框、锁状态和编号预测。两个 bbox 均为归一化的
            `[cx, cy, w, h]`，训练时需要与同格式标签计算损失。
        """
        detection_features, ocr_features = self.backbone(images)
        # detection_features: [B, 256, H16, W16]，示例：[B, 256, 34, 60]
        # ocr_features: [B, 128, H8, W8]，示例：[B, 128, 68, 120]
        src = self.input_projection(detection_features)
        # src: [B, 128, H16, W16]，示例：[B, 128, 34, 60]

        if image_mask is None:
            feature_mask = torch.zeros(
                (src.shape[0], src.shape[2], src.shape[3]),
                dtype=torch.bool,
                device=src.device,
            )
        else:
            feature_mask = F.interpolate(
                image_mask[:, None].float(),
                size=src.shape[-2:],
                mode="nearest",
            )[:, 0].bool()

        # position 与 src 同形状：[B, 128, H16, W16]。
        position = self.position_embedding(src, feature_mask)
        # N = H16 × W16；540×960 输入时 N = 34×60 = 2040。
        src_tokens = src.flatten(2).transpose(1, 2)
        pos_tokens = position.flatten(2).transpose(1, 2)
        padding_mask = feature_mask.flatten(1)
        # src_tokens/pos_tokens: [B, N, 128]；padding_mask: [B, N]

        memory = self.encoder(
            src_tokens + pos_tokens,
            src_key_padding_mask=padding_mask,
        )
        # memory: [B, N, 128]，空间 token 数在 Encoder 前后保持不变。

        batch_size = images.shape[0]
        query_pos = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        target = torch.zeros_like(query_pos)
        # query_pos/target: [B, 20, 128]
        query_features = self.decoder(
            target + query_pos,
            memory,
            memory_key_padding_mask=padding_mask,
        )
        # query_features: [B, 20, 128]，每个 query 表示一个潜在完整实体。

        pred_text_boxes = self.text_bbox_head(query_features).sigmoid()
        outputs = {
            # [B, 20, 5]，四种图元类别加一个 no-object 类。
            "pred_logits": self.class_head(query_features),
            # [B, 20, 4]，归一化 [cx, cy, w, h]。
            "pred_symbol_boxes": self.symbol_bbox_head(query_features).sigmoid(),
            # [B, 20, 4]，归一化 [cx, cy, w, h]。
            "pred_text_boxes": pred_text_boxes,
            # [B, 20, 1]，未经过 sigmoid 的锁状态 logits。
            "pred_locked": self.locked_head(query_features),
            # 保留中间特征，便于后续研究双区域交互或可视化。
            "query_features": query_features,
            "ocr_features": ocr_features,
        }
        if recognize_text:
            # 验证和推理阶段根据每个query自己预测的文字框读取局部特征。
            outputs["pred_identifier_logits"] = self.text_recognition_head(
                ocr_features,
                query_features,
                pred_text_boxes,
            )
            # pred_identifier_logits: [B, 20, 24, 11]，最后一类为CTC blank。
        return outputs


def build_model(pretrained_backbone=True):
    """按照当前项目配置构建一个20查询、四分类的 Entity-DETR。"""
    return EntityDETR(
        num_classes=4,
        num_queries=20,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        pretrained_backbone=pretrained_backbone,
    )
