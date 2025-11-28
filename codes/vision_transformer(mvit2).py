# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import trunc_normal_

from mmpretrain.registry import MODELS
from ..utils import (MultiheadAttention, SwiGLUFFNFused, build_norm_layer,
                     resize_pos_embed, to_2tuple)
from .base_backbone import BaseBackbone

def attention_pool(tensor, pool, hw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, hw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    H, W = hw_shape
    tensor = tensor.reshape(B * N, H, W, C).permute(0, 3, 1, 2).contiguous()

    tensor = pool(tensor)

    hw_shape = [tensor.shape[2], tensor.shape[3]]
    L_pooled = tensor.shape[2] * tensor.shape[3]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)

    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, hw_shape


def cal_rel_pos_spatial(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w):
    sp_idx = 1 if has_cls_embed else 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio)
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio)
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn

import numpy as np
class Attention(nn.Module):
    def __init__(self, dim, dim_out, input_size, num_heads=8, qkv_bias=False, kernel_q=(1, 1), kernel_kv=(1, 1), stride_q=(1, 1), stride_kv=(1, 1), norm_layer=nn.LayerNorm, has_cls_embed=True, mode="conv", pool_first=False, rel_pos_spatial=False, rel_pos_zero_init=False, residual_pooling=True):
        super().__init__()
        self.pool_first = pool_first
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        if pool_first:
            self.q = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.k = nn.Linear(dim, dim_out, bias=qkv_bias)
            self.v = nn.Linear(dim, dim_out, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)

        self.proj = nn.Linear(dim_out, dim_out)

        if np.prod(kernel_q) == 1 and np.prod(stride_q) == 1:
            kernel_q = ()
        if np.prod(kernel_kv) == 1 and np.prod(stride_kv) == 1:
            kernel_kv = ()
        self.mode = mode

        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2d if mode == "max" else nn.AvgPool2d
            self.pool_q = pool_op(kernel_q, stride_q, padding_q, ceil_mode=False) if len(kernel_q) > 0 else None
            self.pool_k = pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False) if len(kernel_kv) > 0 else None
            self.pool_v = pool_op(kernel_kv, stride_kv, padding_kv, ceil_mode=False) if len(kernel_kv) > 0 else None
        elif mode == "conv" or mode == "conv_unshared":
            if pool_first:
                dim_conv = dim // num_heads if mode == "conv" else dim
            else:
                dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            self.pool_q = nn.Conv2d(dim_conv, dim_conv, kernel_q, stride=stride_q, padding=padding_q, groups=dim_conv, bias=False) if len(kernel_q) > 0 else None
            self.norm_q = norm_layer(dim_conv) if len(kernel_q) > 0 else None
            self.pool_k = nn.Conv2d(dim_conv, dim_conv, kernel_kv, stride=stride_kv, padding=padding_kv, groups=dim_conv, bias=False) if len(kernel_kv) > 0 else None
            self.norm_k = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
            self.pool_v = nn.Conv2d(dim_conv, dim_conv, kernel_kv, stride=stride_kv, padding=padding_kv, groups=dim_conv, bias=False) if len(kernel_kv) > 0 else None
            self.norm_v = norm_layer(dim_conv) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        self.rel_pos_spatial = rel_pos_spatial
        if self.rel_pos_spatial:
            assert input_size[0] == input_size[1]

            size = input_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

            if not rel_pos_zero_init:
                trunc_normal_(self.rel_pos_h, std=0.02)
                trunc_normal_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x):
        B, N, _ = x.shape

        H,W=16,12

        if self.pool_first:
            if self.mode == "conv_unshared":
                fold_dim = 1
            else:
                fold_dim = self.num_heads
            x = x.reshape(B, N, fold_dim, -1).permute(0, 2, 1, 3)
            q = k = v = x
        else:
            assert self.mode != "conv_unshared"

            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

        q, q_shape = attention_pool(q, self.pool_q, (H, W), has_cls_embed=self.has_cls_embed, norm=getattr(self, "norm_q", None))
        k, k_shape = attention_pool(k, self.pool_k, (H, W), has_cls_embed=self.has_cls_embed, norm=getattr(self, "norm_k", None))
        v, v_shape = attention_pool(v, self.pool_v, (H, W), has_cls_embed=self.has_cls_embed, norm=getattr(self, "norm_v", None))

        if self.pool_first:
            q_N = np.prod(q_shape) + 1 if self.has_cls_embed else np.prod(q_shape)
            k_N = np.prod(k_shape) + 1 if self.has_cls_embed else np.prod(k_shape)
            v_N = np.prod(v_shape) + 1 if self.has_cls_embed else np.prod(v_shape)
        else:
            q_N, k_N, v_N = q.shape[2], k.shape[2], v.shape[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rel_pos_spatial:
            attn = cal_rel_pos_spatial(attn, q, self.has_cls_embed, q_shape, k_shape, self.rel_pos_h, self.rel_pos_w)

        attn = attn.softmax(dim=-1)

        N = q.shape[2]
        out = (attn @ v).transpose(1, 2).reshape(B, N, self.dim_out)
        out = self.proj(out)

        return out


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension
        num_heads (int): Parallel attention heads
        feedforward_channels (int): The hidden dimension for FFNs
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        attn_drop_rate (float): The drop out rate for attention output weights.
            Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Defaults to 2.
        qkv_bias (bool): enable bias for qkv if True. Defaults to True.
        ffn_type (str): Select the type of ffn layers. Defaults to 'origin'.
        act_cfg (dict): The activation config for FFNs.
            Defaults to ``dict(type='GELU')``.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
             embed_dims,
             num_heads,
             feedforward_channels,
             input_size,  # 非默认参数
             layer_scale_init_value=0.,  # 默认参数
             drop_rate=0.,  # 默认参数
             attn_drop_rate=0.,  # 默认参数
             drop_path_rate=0.,  # 默认参数
             num_fcs=2,  # 默认参数
             qkv_bias=True,  # 默认参数
             ffn_type='origin',  # 默认参数
             act_cfg=dict(type='GELU'),  # 默认参数
             norm_cfg=dict(type='LN'),  # 默认参数
             init_cfg=None,  # 默认参数
             
             kernel_q=(1, 1),  # 非默认参数
             kernel_kv=(1, 1),  # 非默认参数
             stride_q=(1, 1),  # 非默认参数
             stride_kv=(1, 1),  # 非默认参数
             has_cls_embed=True,  # 默认参数
             mode="conv",  # 默认参数
             pool_first=False,  # 默认参数
             rel_pos_spatial=False,  # 默认参数
             rel_pos_zero_init=False,  # 默认参数
             residual_pooling=True  # 默认参数
             ):
        super(TransformerEncoderLayer, self).__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.norm1 = build_norm_layer(norm_cfg, self.embed_dims)
        
        self.attn = Attention(
            dim=embed_dims,
            dim_out=embed_dims,
            input_size=input_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            mode=mode,
            pool_first=pool_first,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_zero_init=rel_pos_zero_init,
            residual_pooling=residual_pooling
        )


        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims) 
        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        if ffn_type == 'origin':
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                layer_scale_init_value=layer_scale_init_value)
        elif ffn_type == 'swiglu_fused':
            self.ffn = SwiGLUFFNFused(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                layer_scale_init_value=layer_scale_init_value)
        else:
            raise NotImplementedError

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(TransformerEncoderLayer, self).init_weights()
        for m in self.ffn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = self.ffn(self.ln2(x), identity=x)
        return x


@MODELS.register_module()
class VisionTransformer(BaseBackbone):
    """Vision Transformer.

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture. If use string,
            choose from 'small', 'base', 'large', 'deit-tiny', 'deit-small'
            and 'deit-base'. If use dict, it should have below keys:

            - **embed_dims** (int): The dimensions of embedding.
            - **num_layers** (int): The number of transformer encoder layers.
            - **num_heads** (int): The number of heads in attention modules.
            - **feedforward_channels** (int): The hidden dimensions in
              feedforward modules.

            Defaults to 'base'.
        img_size (int | tuple): The expected input image shape. Because we
            support dynamic input shape, just set the argument to the most
            common input image shape. Defaults to 224.
        patch_size (int | tuple): The patch size in patch embedding.
            Defaults to 16.
        in_channels (int): The num of input channels. Defaults to 3.
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        qkv_bias (bool): Whether to add bias for qkv in attention modules.
            Defaults to True.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        out_type (str): The type of output features. Please choose from

            - ``"cls_token"``: The class token tensor with shape (B, C).
            - ``"featmap"``: The feature map tensor from the patch tokens
              with shape (B, C, H, W).
            - ``"avg_featmap"``: The global averaged feature map tensor
              with shape (B, C).
            - ``"raw"``: The raw feature tensor includes patch tokens and
              class tokens with shape (B, L, C).

            Defaults to ``"cls_token"``.
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Defaults to True.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Defaults to -1.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        layer_scale_init_value (float or torch.Tensor): Init value of layer
            scale. Defaults to 0.
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                # The same as the implementation in MAE
                # <https://arxiv.org/abs/2111.06377>
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120
            }),
        **dict.fromkeys(
            ['eva-g', 'eva-giant'],
            {
                # The implementation in EVA
                # <https://arxiv.org/abs/2211.07636>
                'embed_dims': 1408,
                'num_layers': 40,
                'num_heads': 16,
                'feedforward_channels': 6144
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small', 'dinov2-s', 'dinov2-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
        **dict.fromkeys(
            ['dinov2-g', 'dinov2-giant'], {
                'embed_dims': 1536,
                'num_layers': 40,
                'num_heads': 24,
                'feedforward_channels': 6144
            }),
    }
    num_extra_tokens = 1  # class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}

    def __init__(self,
                 arch='base',
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 qkv_bias=True,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 out_type='cls_token',
                 with_cls_token=True,
                 frozen_stages=-1,
                 interpolate_mode='bicubic',
                 layer_scale_init_value=0.,
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 pre_norm=False,
                 init_cfg=None,
                 kernel_q=(1, 1),  # 添加这些参数
                 kernel_kv=(1, 1),
                 stride_q=(1, 1),
                 stride_kv=(1, 1),
                 num_keypoints=10,
                 rel_pos_spatial=False,
                 rel_pos_zero_init=False,
                 residual_pooling=True,
                 has_cls_embed=True,
                 mode="conv",
                 pool_first=False):
        super(VisionTransformer, self).__init__(init_cfg)
        
        
        input_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.img_size = to_2tuple(img_size)

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=not pre_norm,  # disable bias if pre_norm is used(e.g., CLIP)
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        # Set cls token
        self.with_cls_token = with_cls_token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        elif out_type != 'cls_token':
            self.cls_token = None
            self.num_extra_tokens = 0
        else:
            raise ValueError(
                'with_cls_token must be True when `out_type="cls_token"`.')

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens,
                        self.embed_dims))
        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.num_layers)

        self.input_size = input_size
        self.kernel_q = kernel_q
        self.kernel_kv = kernel_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv
        self.num_keypoints = num_keypoints
        self.qkv_bias = qkv_bias
        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_zero_init = rel_pos_zero_init
        self.residual_pooling = residual_pooling
        self.has_cls_embed = has_cls_embed
        self.mode = mode
        self.pool_first = pool_first
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_rate = drop_rate
        # self.attn_drop_rate = attn_drop_rate
        # self.drop_path_rate = drop_path_rate
        # self.num_fcs = num_fcs
        # self.ffn_type = ffn_type
        # self.act_cfg = act_cfg
        # self.norm_cfg = norm_cfg
        # self.init_cfg = init_cfg

        # Assuming you have self.num_layers and self.arch_settings defined elsewhere
        self.layers = nn.ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.arch_settings['embed_dims'],
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.arch_settings['feedforward_channels'],
                kernel_q=self.kernel_q,
                kernel_kv=self.kernel_kv,
                stride_q=self.stride_q,
                stride_kv=self.stride_kv,
                input_size = (img_size[0] // patch_size, img_size[1] // patch_size),
                qkv_bias=self.qkv_bias,
                rel_pos_spatial=self.rel_pos_spatial,
                rel_pos_zero_init=self.rel_pos_zero_init,
                residual_pooling=self.residual_pooling,
                has_cls_embed=self.has_cls_embed,
                mode=self.mode,
                pool_first=self.pool_first,
                layer_scale_init_value=self.layer_scale_init_value,
                drop_rate=drop_rate,
                attn_drop_rate=0.,
                drop_path_rate=dpr[i],
                norm_cfg=norm_cfg
            )
            _layer_cfg.update(layer_cfgs[i])
            
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.frozen_stages = frozen_stages
        if pre_norm:
            self.pre_norm = build_norm_layer(norm_cfg, self.embed_dims)
        else:
            self.pre_norm = nn.Identity()

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)
        if self.out_type == 'avg_featmap':
            self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(VisionTransformer, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape
        if (not self.with_cls_token
                and ckpt_pos_embed_shape[1] == self.pos_embed.shape[1] + 1):
            # Remove cls token from state dict if it's not used.
            state_dict[name] = state_dict[name][:, 1:]
            ckpt_pos_embed_shape = state_dict[name].shape

        if self.pos_embed.shape != ckpt_pos_embed_shape:
            from mmengine.logging import MMLogger
            logger = MMLogger.get_current_instance()
            logger.info(
                f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                f'to {self.pos_embed.shape}.')

            ckpt_pos_embed_shape = to_2tuple(
                int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
            pos_embed_shape = self.patch_embed.init_out_size

            state_dict[name] = resize_pos_embed(state_dict[name],
                                                ckpt_pos_embed_shape,
                                                pos_embed_shape,
                                                self.interpolate_mode,
                                                self.num_extra_tokens)

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze pre-norm
        for param in self.pre_norm.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

            if self.out_type == 'avg_featmap':
                self.ln2.eval()
                for param in self.ln2.parameters():
                    param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)

    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens:]
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            return self.ln2(patch_token.mean(dim=1))

    def get_layer_depth(self, param_name: str, prefix: str = ''):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.
            prefix (str): The prefix for the parameter.
                Defaults to an empty string.

        Returns:
            Tuple[int, int]: The layer-wise depth and the num of layers.

        Note:
            The first depth is the stem module (``layer_depth=0``), and the
            last depth is the subsequent module (``layer_depth=num_layers-1``)
        """
        num_layers = self.num_layers + 2

        if not param_name.startswith(prefix):
            # For subsequent module like head
            return num_layers - 1, num_layers

        param_name = param_name[len(prefix):]

        if param_name in ('cls_token', 'pos_embed'):
            layer_depth = 0
        elif param_name.startswith('patch_embed'):
            layer_depth = 0
        elif param_name.startswith('layers'):
            layer_id = int(param_name.split('.')[1])
            layer_depth = layer_id + 1
        else:
            layer_depth = num_layers - 1

        return layer_depth, num_layers
