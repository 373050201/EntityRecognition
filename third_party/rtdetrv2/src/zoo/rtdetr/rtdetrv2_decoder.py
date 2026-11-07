"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math 
import copy 
import functools
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init 
from typing import List

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func_v2, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob

from ...core import register

__all__ = ['RTDETRTransformerv2']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(
        self, 
        embed_dim=256, 
        num_heads=8, 
        num_levels=4, 
        num_points=4, 
        method='default',
        offset_scale=0.5,
    ):
        """Multi-Scale Deformable Attention
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels, ''
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        
        num_points_scale = [1/n for n in num_points_list for _ in range(n)]
        self.register_buffer('num_points_scale', torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = functools.partial(deformable_attention_core_func_v2, method=self.method) 

        self._reset_parameters()

        if method == 'discrete':
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.concat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)


    def forward(self,
                query: torch.Tensor,
                reference_points: torch.Tensor,
                value: torch.Tensor,
                value_spatial_shapes: List[int],
                value_mask: torch.Tensor=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value = value * value_mask.to(value.dtype).unsqueeze(-1)

        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets: torch.Tensor = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

        attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            # reference_points [8, 480, None, 1,  4]
            # sampling_offsets [8, 480, 8,    12, 2]
            num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights, self.num_points_list)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation='relu',
                 n_levels=4,
                 n_points=4,
                 cross_attn_method='default',
                 region_interaction_mode='none',
                 region_interaction_init_scale=1e-3):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # 文字分支：围绕独立文字参考框采样文字区域特征。
        self.text_cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.text_dropout = nn.Dropout(dropout)
        self.text_norm = nn.LayerNorm(d_model)

        # V1-C/V1-D：控制关闭、双向、图元到文字三种区域交互模式。
        valid_interaction_modes = {'none', 'bidirectional', 'symbol_to_text'}
        if region_interaction_mode not in valid_interaction_modes:
            raise ValueError(
                f"Unsupported region_interaction_mode: {region_interaction_mode}"
            )
        self.region_interaction_mode = region_interaction_mode
        if region_interaction_mode != 'none':
            self.region_interaction_norm = nn.LayerNorm(d_model)
            self.region_interaction_attn = nn.MultiheadAttention(
                d_model,
                n_head,
                dropout=dropout,
                batch_first=True,
            )
            self.region_interaction_dropout = nn.Dropout(dropout)
            # `[1, 2, 1]`分别控制图元token和文字token的融合强度，小值起步保护原生路径。
            self.region_interaction_scale = nn.Parameter(
                torch.full((1, 2, 1), region_interaction_init_scale)
            )

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = get_activation(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def _interact_entity_regions(
        self,
        target,
        text_target,
        num_regular_queries,
    ):
        """只在正式entity query内部交互图元、文字区域特征。"""
        if self.region_interaction_mode == 'none':
            return target, text_target
        if num_regular_queries <= 0 or num_regular_queries > target.shape[1]:
            raise ValueError("num_regular_queries is inconsistent with query shape")

        # 去噪query位于前部且没有文字监督；只取末尾正式query参与区域交互。
        regular_target = target[:, -num_regular_queries:]
        regular_text_target = text_target[:, -num_regular_queries:]

        # `[B, Q, 2, C] -> [B*Q, 2, C]`：每组注意力只包含同一实体的两个区域。
        region_tokens = torch.stack(
            [regular_target, regular_text_target],
            dim=2,
        )
        batch_size, query_count, _, hidden_dim = region_tokens.shape
        region_tokens = region_tokens.reshape(
            batch_size * query_count,
            2,
            hidden_dim,
        )
        normalized_tokens = self.region_interaction_norm(region_tokens)
        if self.region_interaction_mode == 'bidirectional':
            interaction, _ = self.region_interaction_attn(
                normalized_tokens,
                normalized_tokens,
                normalized_tokens,
                need_weights=False,
            )
            region_tokens = region_tokens + (
                self.region_interaction_scale
                * self.region_interaction_dropout(interaction)
            )
        else:
            # V1-D：文字token在自身和所属图元之间选择信息，只更新文字分支。
            text_query = normalized_tokens[:, 1:2]
            text_interaction, _ = self.region_interaction_attn(
                text_query,
                normalized_tokens,
                normalized_tokens,
                need_weights=False,
            )
            fused_text_token = region_tokens[:, 1:2] + (
                self.region_interaction_scale[:, 1:2]
                * self.region_interaction_dropout(text_interaction)
            )
            region_tokens = torch.cat(
                [region_tokens[:, :1], fused_text_token],
                dim=1,
            )
        region_tokens = region_tokens.reshape(
            batch_size,
            query_count,
            2,
            hidden_dim,
        )
        fused_target = region_tokens[:, :, 0]
        fused_text_target = region_tokens[:, :, 1]

        # 保留前部去噪query原值，避免无文字监督的特征反向污染图元去噪训练。
        target = torch.cat(
            [target[:, :-num_regular_queries], fused_target],
            dim=1,
        )
        text_target = torch.cat(
            [text_target[:, :-num_regular_queries], fused_text_target],
            dim=1,
        )
        return target, text_target

    def forward(self,
                target,
                reference_points,
                text_reference_points,
                memory,
                memory_spatial_shapes,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None,
                text_query_pos_embed=None,
                num_regular_queries=None,
                primary_only=False):
        # self attention
        q = k = self.with_pos_embed(target, query_pos_embed)

        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask, need_weights=False)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # cross attention
        target2 = self.cross_attn(\
            self.with_pos_embed(target, query_pos_embed), 
            reference_points, 
            memory, 
            memory_spatial_shapes, 
            memory_mask)
        target = target + self.dropout2(target2)
        target = self.norm2(target)

        # ffn
        target2 = self.forward_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target)

        if primary_only:
            return target, None

        # `[B, Q_total, C]`：使用图元路径更新后的query在文字参考框附近独立取样。
        text_target = self.text_cross_attn(
            self.with_pos_embed(target, text_query_pos_embed),
            text_reference_points,
            memory,
            memory_spatial_shapes,
            memory_mask)
        text_target = target + self.text_dropout(text_target)
        text_target = self.text_norm(text_target)

        # 按配置在同一正式entity query内交互图元、文字区域特征。
        if num_regular_queries is None:
            num_regular_queries = target.shape[1]
        target, text_target = self._interact_entity_regions(
            target,
            text_target,
            num_regular_queries,
        )

        return target, text_target


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        num_layers,
        eval_idx=-1,
        num_queries=None,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                target,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                bbox_head,
                text_bbox_head,
                score_head,
                query_pos_head,
                text_ref_points_unact,
                attn_mask=None,
                memory_mask=None,
                primary_only=False):
        dec_out_bboxes = []
        dec_out_text_bboxes = []
        dec_out_logits = []
        dec_out_features = []
        dec_out_text_features = []
        ref_points_detach = F.sigmoid(ref_points_unact)
        text_ref_points_detach = (
            F.sigmoid(text_ref_points_unact)
            if not primary_only else None
        )

        output = target
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)
            text_ref_points_input = (
                text_ref_points_detach.unsqueeze(2)
                if not primary_only else None
            )
            text_query_pos_embed = (
                query_pos_head(text_ref_points_detach)
                if not primary_only else None
            )

            output, text_output = layer(
                output,
                ref_points_input,
                text_ref_points_input,
                memory,
                memory_spatial_shapes,
                attn_mask,
                memory_mask,
                query_pos_embed,
                text_query_pos_embed,
                num_regular_queries=self.num_queries,
                primary_only=primary_only)
            # 实体模型的文字框分支需要各层 query 特征执行逐层迭代回归。
            # 训练和推理阶段都保留，最终检测输出仍使用 eval_idx 对应层。
            dec_out_features.append(output)
            if not primary_only:
                dec_out_text_features.append(text_output)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))
            if not primary_only:
                inter_text_ref_bbox = F.sigmoid(
                    text_bbox_head[i](text_output)
                    + inverse_sigmoid(text_ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                    if not primary_only:
                        dec_out_text_bboxes.append(inter_text_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
                    if not primary_only:
                        dec_out_text_bboxes.append(F.sigmoid(
                            text_bbox_head[i](text_output)
                            + inverse_sigmoid(text_ref_points)))

            else:
                # 验证阶段同样保留各层文字框，使文字辅助损失口径与原实现一致。
                if not primary_only:
                    dec_out_text_bboxes.append(inter_text_ref_bbox)
                if i == self.eval_idx:
                    dec_out_logits.append(score_head[i](output))
                    dec_out_bboxes.append(inter_ref_bbox)
                    break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach()
            if not primary_only:
                text_ref_points = inter_text_ref_bbox
                text_ref_points_detach = inter_text_ref_bbox.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_text_bboxes) if not primary_only else None,
            torch.stack(dec_out_logits),
            torch.stack(dec_out_features),
            torch.stack(dec_out_text_features) if not primary_only else None,
        )


@register()
class RTDETRTransformerv2(nn.Module):
    __share__ = ['num_classes', 'eval_spatial_size']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 feat_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 num_levels=3,
                 num_points=4,
                 nhead=8,
                 num_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learn_query_content=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2, 
                 aux_loss=True, 
                 cross_attn_method='default', 
                 query_select_method='default',
                 text_reference_offset_scale=5.0,
                 text_reference_jitter_scale=0.2,
                 text_first_layer_sampling_scale=5.0,
                 text_refine_sampling_scale=0.5,
                 region_interaction_mode='none',
                 region_interaction_init_scale=1e-3,
                 primary_only=False):
        super().__init__()
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss
        self.text_reference_offset_scale = text_reference_offset_scale
        self.text_reference_jitter_scale = text_reference_jitter_scale
        self.region_interaction_mode = region_interaction_mode
        self.primary_only = primary_only

        if text_reference_offset_scale <= 0:
            raise ValueError("text_reference_offset_scale must be greater than 0")
        if text_reference_jitter_scale < 0:
            raise ValueError("text_reference_jitter_scale must be non-negative")
        if text_first_layer_sampling_scale <= 0:
            raise ValueError("text_first_layer_sampling_scale must be greater than 0")
        if text_refine_sampling_scale <= 0:
            raise ValueError("text_refine_sampling_scale must be greater than 0")
        if region_interaction_init_scale < 0:
            raise ValueError("region_interaction_init_scale must be non-negative")
        if region_interaction_mode not in ('none', 'bidirectional', 'symbol_to_text'):
            raise ValueError(
                f"Unsupported region_interaction_mode: {region_interaction_mode}"
            )

        assert query_select_method in ('default', 'one2many', 'agnostic'), ''
        assert cross_attn_method in ('default', 'discrete'), ''
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, \
            activation, num_levels, num_points, cross_attn_method=cross_attn_method, \
            region_interaction_mode=region_interaction_mode, \
            region_interaction_init_scale=region_interaction_init_scale)
        self.decoder = TransformerDecoder(
            hidden_dim,
            decoder_layer,
            num_layers,
            eval_idx,
            num_queries=num_queries,
        )
        for layer_index, layer in enumerate(self.decoder.layers):
            layer.text_cross_attn.offset_scale = (
                text_first_layer_sampling_scale
                if layer_index == 0
                else text_refine_sampling_scale
            )

        # denoising
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if num_denoising > 0: 
            self.denoising_class_embed = nn.Embedding(num_classes+1, hidden_dim, padding_idx=num_classes)
            init.normal_(self.denoising_class_embed.weight[:-1])

        # decoder embedding
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, 2)
        # 初始文字框由entity query相对图元参考框确定性预测。
        self.text_ref_point_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # if num_select_queries != self.num_queries:
        #     layer = TransformerEncoderLayer(hidden_dim, nhead, dim_feedforward, activation='gelu')
        #     self.encoder = TransformerEncoder(layer, 1)

        self.enc_output = nn.Sequential(OrderedDict([
            ('proj', nn.Linear(hidden_dim, hidden_dim)),
            ('norm', nn.LayerNorm(hidden_dim,)),
        ]))

        if query_select_method == 'agnostic':
            self.enc_score_head = nn.Linear(hidden_dim, 1)
        else:
            self.enc_score_head = nn.Linear(hidden_dim, num_classes)

        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for _ in range(num_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
        ])
        self.dec_text_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer('anchors', anchors)
            self.register_buffer('valid_mask', valid_mask)

        self._reset_parameters()
        
    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for _cls, _reg in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(_cls.bias, bias)
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)

        # 零残差初始化使文字参考框先从图元参考框稳定起步，再由训练学习相对位置。
        init.constant_(self.text_ref_point_head.layers[-1].weight, 0)
        init.constant_(self.text_ref_point_head.layers[-1].bias, 0)
        for _reg in self.dec_text_bbox_head:
            init.constant_(_reg.layers[-1].weight, 0)
            init.constant_(_reg.layers[-1].bias, 0)
        
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)
        for m in self.input_proj:
            init.xavier_uniform_(m[0].weight)

    def initialize_text_attention_from_symbol(self):
        """使用已加载的图元交叉注意力参数初始化文字采样分支。"""
        for layer in self.decoder.layers:
            layer.text_cross_attn.load_state_dict(layer.cross_attn.state_dict())

    def _build_initial_text_reference(self, query, symbol_ref_points_unact):
        """预测文字初始框，并在训练时扰动正式entity query的文字中心。"""
        symbol_reference = F.sigmoid(symbol_ref_points_unact)
        text_residual = self.text_ref_point_head(query)

        # 中心偏移限制在图元尺度定义的连续邻域内，不预设上下左右方向。
        symbol_radius = symbol_reference[..., 2:].amax(dim=-1, keepdim=True)
        text_center = symbol_reference[..., :2] + (
            torch.tanh(text_residual[..., :2])
            * symbol_radius
            * self.text_reference_offset_scale
        )

        # `[B, Q_total, 2]`：V1-B只在训练时扰动末尾正式entity query。
        # 去噪query排列在前且没有文字监督，因此掩码为0；验证和推理不进入此分支。
        if self.training and self.text_reference_jitter_scale > 0:
            query_indices = torch.arange(
                text_center.shape[1],
                device=text_center.device,
            )
            regular_query_start = text_center.shape[1] - self.num_queries
            regular_query_mask = (
                query_indices >= regular_query_start
            ).to(text_center.dtype).view(1, -1, 1)
            center_jitter = (
                torch.rand_like(text_center) * 2 - 1
            ) * symbol_radius * self.text_reference_jitter_scale
            text_center = text_center + center_jitter * regular_query_mask

        text_center = text_center.clamp(self.eps, 1 - self.eps)

        # 宽高相对图元初始尺寸独立学习，零残差时从图元尺寸稳定起步。
        text_size = F.sigmoid(
            inverse_sigmoid(symbol_reference[..., 2:])
            + text_residual[..., 2:]
        )
        text_reference = torch.concat([text_center, text_size], dim=-1)
        return inverse_sigmoid(text_reference)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)), 
                    ('norm', nn.BatchNorm2d(self.hidden_dim,))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, feats: List[torch.Tensor]):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.concat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.concat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask


    def _get_decoder_input(self,
                           memory: torch.Tensor,
                           spatial_shapes,
                           denoising_logits=None,
                           denoising_bbox_unact=None):

        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask

        # memory = torch.where(valid_mask, memory, 0)
        # TODO fix type error for onnx export 
        memory = valid_mask.to(memory.dtype) * memory  

        output_memory :torch.Tensor = self.enc_output(memory)
        enc_outputs_logits :torch.Tensor = self.enc_score_head(output_memory)
        enc_outputs_coord_unact :torch.Tensor = self.enc_bbox_head(output_memory) + anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        enc_topk_memory, enc_topk_logits, enc_topk_bbox_unact = \
            self._select_topk(output_memory, enc_outputs_logits, enc_outputs_coord_unact, self.num_queries)
            
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        # if self.num_select_queries != self.num_queries:            
        #     raise NotImplementedError('')

        if self.learn_query_content:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()
            
        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()
        
        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.concat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.concat([denoising_logits, content], dim=1)
        
        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory: torch.Tensor, outputs_logits: torch.Tensor, outputs_coords_unact: torch.Tensor, topk: int):
        if self.query_select_method == 'default':
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)

        elif self.query_select_method == 'one2many':
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes

        elif self.query_select_method == 'agnostic':
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        
        topk_ind: torch.Tensor

        topk_coords = outputs_coords_unact.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_coords_unact.shape[-1]))
        
        topk_logits = outputs_logits.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
        
        topk_memory = memory.gather(dim=1, \
            index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1]))

        return topk_memory, topk_logits, topk_coords


    def forward(self, feats, targets=None):
        # input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(feats)
        
        # prepare denoising training
        if self.training and self.num_denoising > 0:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                    self.num_classes, 
                    self.num_queries, 
                    self.denoising_class_embed, 
                    num_denoising=self.num_denoising, 
                    label_noise_ratio=self.label_noise_ratio, 
                    box_noise_scale=self.box_noise_scale, )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = \
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)

        # `[B, Q_total, 4]`：先确定性预测文字初始框，再按V1-B规则处理。
        # 训练时仅正式entity query加入中心扰动，验证和推理仍为确定性初始化。
        init_text_ref_points_unact = (
            self._build_initial_text_reference(
                init_ref_contents,
                init_ref_points_unact,
            )
            if not self.primary_only else None
        )

        # decoder
        (
            out_bboxes,
            out_text_bboxes,
            out_logits,
            out_query_features,
            out_text_query_features,
        ) = self.decoder(
            init_ref_contents,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            self.dec_bbox_head,
            self.dec_text_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            init_text_ref_points_unact,
            attn_mask=attn_mask,
            primary_only=self.primary_only)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            if not self.primary_only:
                _, out_text_bboxes = torch.split(out_text_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            _, out_query_features = torch.split(
                out_query_features,
                dn_meta['dn_num_split'],
                dim=2,
            )
            if not self.primary_only:
                _, out_text_query_features = torch.split(
                    out_text_query_features,
                    dn_meta['dn_num_split'],
                    dim=2,
                )

        out = {
            'pred_logits': out_logits[-1],
            'pred_boxes': out_bboxes[-1],
            'query_features': out_query_features[-1],
            'decoder_query_features': out_query_features,
        }
        if not self.primary_only:
            out.update({
                'pred_text_boxes': out_text_bboxes[-1],
                'text_query_features': out_text_query_features[-1],
                'decoder_text_query_features': out_text_query_features,
                'text_aux_outputs': [
                    {'pred_text_boxes': text_boxes}
                    for text_boxes in out_text_bboxes[:-1]
                ],
            })

        if self.training and self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['enc_aux_outputs'] = self._set_aux_loss(enc_topk_logits_list, enc_topk_bboxes_list)
            out['enc_meta'] = {'class_agnostic': self.query_select_method == 'agnostic'}

            if dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]
