# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import drop_path, to_2tuple, trunc_normal_

from ..builder import BACKBONES
from .base_backbone import BaseBackbone

def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """
    Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token
        dimension for the original embeddings.
    Args:
        abs_pos (Tensor): absolute positional embeddings with (1, num_position, C).
        has_cls_token (bool): If true, has 1 embedding in abs_pos for cls token.
        hw (Tuple): size of input image tokens.

    Returns:
        Absolute positional embeddings after processing with shape (1, H, W, C)
    """
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).reshape(B, -1, C)

    else:
        new_abs_pos = abs_pos
    
    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# def batch_index_select(x, idx):
#     if len(x.size()) == 3:
#         B, N, C = x.size()
#         N_new = idx.size(1)
#         offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
#         idx = idx + offset
#         out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
#         return out
#     elif len(x.size()) == 2:
#         B, N = x.size()
#         N_new = idx.size(1)
#         offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
#         idx = idx + offset
#         out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
#         return out
#     else:
#         raise NotImplementedError

# def get_index(idx, patch_shape_src, patch_shape_des):
#     '''
#     get index of fine stage corresponding to coarse stage 
#     '''
#     h1, w1 = patch_shape_src
#     h2, w2 = patch_shape_des
#     hs = h2 // h1
#     ws = w2 // w1            
    
#     j = idx % w1
#     i = torch.div(idx, w1, rounding_mode='floor')
    
#     idx = i * hs * w2 + j * ws
    
#     idxs = []
#     for i in range(hs):
#         for j in range(ws):
#             idxs.append(idx + i * w2 + j)
    
#     return torch.cat(idxs, dim=1)

# class QualityPredictor(nn.Module):
#     def __init__(self, embed=768, drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, 
#                  sigmoid=False, qp_abs=False) -> None:
#         super().__init__()
        
#         self.mlp = nn.Sequential(
#             nn.Linear(embed, embed),
#             act_layer(),
#             nn.Dropout(drop),
#             nn.Linear(embed, embed),
#             act_layer(),
#             nn.Dropout(drop),
#             nn.Linear(embed, 2 if qp_abs else 1),
#             nn.Softmax(dim=-1) if qp_abs else nn.Sigmoid() if sigmoid else act_layer(),
#         )
#         self.norm = norm_layer(embed)
        
#     def forward(self, x: torch.Tensor):
#         x = x.mean(dim=1)
#         x = self.norm(x)
#         x = self.mlp(x)
#         return x
    
    

# class Attention(nn.Module):
#     def __init__(
#             self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#             proj_drop=0., attn_head_dim=None):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dim = dim

#         if attn_head_dim is not None:
#             head_dim = attn_head_dim
#         all_head_dim = head_dim * self.num_heads

#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(all_head_dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x,H,W):
#         B, N, C = x.shape
#         qkv = self.qkv(x)
#         qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

#         # Calculate covariance-based attention
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)
        
#         cov = q @ k.transpose(-2, -1)  # Covariance matrix
        
#         attn = cov.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x

    

# class Attention(nn.Module):
#     def __init__(
#             self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#             proj_drop=0., attn_head_dim=None, side_dwconv=3, pool_type='avg',num_keypoints=16):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dim = dim
        
#         if attn_head_dim is not None:
#             head_dim = attn_head_dim
#         all_head_dim = head_dim * self.num_heads

#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(all_head_dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else lambda x: torch.zeros_like(x)
#         self.lepe1 = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else lambda x: torch.zeros_like(x)
        
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         # Define the pooling layer
#         if pool_type == 'avg':
#             self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         elif pool_type == 'max':
#             self.pool = nn.AdaptiveMaxPool2d((1, 1))
#         else:
#             raise ValueError("pool_type must be either 'avg' or 'max'")

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         qkv = self.qkv(x)
#         qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#         # Standard Attention
#         q_norm = F.normalize(q, dim=-1)
#         k_norm = F.normalize(k, dim=-1)
#         cov = q_norm @ k_norm.transpose(-2, -1) * self.temperature  # Covariance matrix
#         attn = cov.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x_standard = (attn @ v).transpose(1, 2).reshape(B, N, -1)
#         x_standard = self.proj(x_standard)
#         x_standard = self.proj_drop(x_standard)

#         # Class Attention
#         qc = q[:, :, 0:1]  # CLS token
#         attn_cls = (qc * k).sum(dim=-1) * self.scale
#         attn_cls = attn_cls.softmax(dim=-1)
#         attn_cls = self.attn_drop(attn_cls)
#         cls_tkn = (attn_cls.unsqueeze(2) @ v).transpose(1, 2).reshape(B, 1, C)
#         cls_tkn = self.proj(cls_tkn)
#         cls_tkn = self.proj_drop(cls_tkn)

#         # Apply Local Perception Enhancement (LePE)
#         lepe = x.transpose(1, 2).view(B, C, H, W)
#         lepe = self.lepe(lepe)
#         lepe = self.lepe1(lepe)

#         # Apply pooling to LePE
#         lepe = self.pool(lepe).view(B, C, 1).transpose(1, 2)

#         x = x + lepe

#         x = torch.cat([cls_tkn, x_standard[:, 1:]], dim=1)

#         return x
# class Attention(nn.Module):
#     def __init__(
#             self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#             proj_drop=0., attn_head_dim=None, side_dwconv=3, pool_type='avg', num_keypoints=16):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dim = dim
        
#         if attn_head_dim is not None:
#             head_dim = attn_head_dim
#         all_head_dim = head_dim * self.num_heads

#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(all_head_dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else lambda x: torch.zeros_like(x)
#         self.lepe1 = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else lambda x: torch.zeros_like(x)
        
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         # Define the pooling layer
#         if pool_type == 'avg':
#             self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         elif pool_type == 'max':
#             self.pool = nn.AdaptiveMaxPool2d((1, 1))
#         else:
#             raise ValueError("pool_type must be either 'avg' or 'max'")

#         # Define keypoint tokens
#         self.keypoint_tokens = nn.Parameter(torch.zeros(1, num_keypoints, dim))

#     def forward(self, x, H, W):
#         B, N, C = x.shape
        
#         # Append keypoint tokens to input
#         keypoint_tokens = self.keypoint_tokens.expand(B, -1, -1)
#         x = torch.cat((x, keypoint_tokens), dim=1)
        
#         qkv = self.qkv(x)
#         qkv = qkv.reshape(B, N + keypoint_tokens.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#         # Standard Attention
#         q_norm = F.normalize(q, dim=-1)
#         k_norm = F.normalize(k, dim=-1)
#         cov = q_norm @ k_norm.transpose(-2, -1) * self.temperature  # Covariance matrix
#         attn = cov.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#         x_standard = (attn @ v).transpose(1, 2).reshape(B, N + keypoint_tokens.shape[1], -1)
#         x_standard = self.proj(x_standard)
#         x_standard = self.proj_drop(x_standard)

#         # Apply Local Perception Enhancement (LePE)
#         lepe = x[:, :-keypoint_tokens.shape[1]].transpose(1, 2).view(B, C, H, W)  # exclude keypoint tokens
#         lepe = self.lepe(lepe)
#         lepe = self.lepe1(lepe)

#         # Apply pooling to LePE
#         lepe = self.pool(lepe).view(B, C, 1).transpose(1, 2)

#         x = x[:, :-keypoint_tokens.shape[1]]  # exclude keypoint tokens
#         x = x + lepe

#         x = torch.cat([x_standard[:, :-keypoint_tokens.shape[1]]], dim=1)

#         return x
    
# class Attention(nn.Module):
#     def __init__(
#             self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#             proj_drop=0., attn_head_dim=None, window_size=7, topk=4):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dim = dim
#         self.window_size = window_size
#         self.topk = topk

#         if attn_head_dim is not None:
#             head_dim = attn_head_dim
#         all_head_dim = head_dim * self.num_heads

#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(all_head_dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x)
#         qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         # Normalize
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         # Local Attention (within windows)
#         local_attn = []
#         window_size = min(self.window_size, N)
#         for i in range(0, N, window_size):
#             end = min(i + window_size, N)
#             q_win, k_win, v_win = q[:, :, i:end], k[:, :, i:end], v[:, :, i:end]
#             cov_win = q_win @ k_win.transpose(-2, -1)
#             local_attn_win = cov_win.softmax(dim=-1)
#             local_attn_win = self.attn_drop(local_attn_win)
#             local_attn.append((local_attn_win @ v_win).transpose(1, 2).reshape(B, end - i, -1))
#         local_attn = torch.cat(local_attn, dim=1)

#         # Global Attention (across windows)
#         cov_global = q @ k.transpose(-2, -1)
#         global_attn = cov_global.softmax(dim=-1)
#         global_attn = self.attn_drop(global_attn)

#         # Combine Local and Global Attention
#         combined_attn = (local_attn + (global_attn @ v).transpose(1, 2).reshape(B, N, -1)) / 2
#         x = self.proj(combined_attn)
#         x = self.proj_drop(x)

#         return x



class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Standard Self-Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x_standard = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x_standard = self.proj(x_standard)
        x_standard = self.proj_drop(x_standard)

        # Class Attention
        qc = q[:, :, 0:1]  # CLS token
        attn_cls = (qc * k).sum(dim=-1) * self.scale
        attn_cls = attn_cls.softmax(dim=-1)
        attn_cls = self.attn_drop(attn_cls)
        cls_tkn = (attn_cls.unsqueeze(2) @ v).transpose(1, 2).reshape(B, 1, C)
        cls_tkn = self.proj(cls_tkn)
        cls_tkn = self.proj_drop(cls_tkn)

        # Combine the results from standard attention and class attention
        x = torch.cat([cls_tkn, x_standard[:, 1:]], dim=1)

        return x




# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None, side_dwconv=1):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dim = dim

#         if attn_head_dim is not None:
#             head_dim = attn_head_dim
#         all_head_dim = head_dim * self.num_heads

#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(all_head_dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv // 2, groups=dim) if side_dwconv > 0 else lambda x: torch.zeros_like(x)

#     def forward(self, x, H, W):
#         B, N, C = x.shape

#         # Split x into class token and the rest
#         cls_token = x[:, :1, :]  # Class token
#         x = x[:, 1:, :]  # Patches

#         qkv = self.qkv(x)
#         qkv = qkv.reshape(B, N-1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         # Normalize
#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         # Apply Local Perception Enhancement (LePE)
#         lepe = x.transpose(1, 2).view(B, C, H, W)
#         lepe = self.lepe(lepe)
#         lepe = lepe.view(B, C, N-1).transpose(1, 2)

#         # Calculate attention for patches
#         cov = q @ k.transpose(-2, -1)
#         attn = cov.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x_patches = (attn @ v).transpose(1, 2).reshape(B, N-1, -1)
#         x_patches = x_patches + lepe
#         x_patches = self.proj(x_patches)
#         x_patches = self.proj_drop(x_patches)

#         # Calculate attention for class token
#         cls_qkv = self.qkv(cls_token)
#         cls_qkv = cls_qkv.reshape(B, 1, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         cls_q, cls_k, cls_v = cls_qkv[0], cls_qkv[1], cls_qkv[2]

#         cls_attn = (cls_q @ k.transpose(-2, -1)).softmax(dim=-1)
#         cls_attn = self.attn_drop(cls_attn)

#         cls_token = (cls_attn @ v).transpose(1, 2).reshape(B, 1, -1)
#         cls_token = self.proj(cls_token)
#         cls_token = self.proj_drop(cls_token)

#         # Concatenate class token with the rest of the tokens
#         x = torch.cat((cls_token, x_patches), dim=1)

#         return x

# class Attention(nn.Module):
#     def __init__(
#             self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#             proj_drop=0., attn_head_dim=None,):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.dim = dim

#         if attn_head_dim is not None:
#             head_dim = attn_head_dim
#         all_head_dim = head_dim * self.num_heads

#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(all_head_dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x,H,W):
#         B, N, C = x.shape
#         qkv = self.qkv(x)
#         qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))

#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
#         x = self.proj(x)
#         x = self.proj_drop(x)

#         return x

class LPI(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU):
        super(LPI, self).__init__()
        self.conv1_3x3 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1)
        self.bn1_3x3 = nn.BatchNorm2d(in_features)

        self.conv2_3x3_dilated = nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bn2_3x3_dilated = nn.BatchNorm2d(in_features)

        self.act = act_layer()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        x1 = self.act(self.bn1_3x3(self.conv1_3x3(x)))
        x2 = self.act(self.bn2_3x3_dilated(self.conv2_3x3_dilated(x)))

        # Combine the results from different convolution layers
        x = x1 + x2 

        x = x.view(B, C, N).transpose(1, 2)
        return x





    
# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm, attn_head_dim=None, eta=1.0, use_attn=True, num_keypoints=10):
#         super().__init__()
#         self.use_attn = use_attn
#         self.num_keypoints = num_keypoints

#         self.norm1 = norm_layer(dim)
#         if use_attn:
#             self.attn = Attention(
#                 dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
#             )

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         self.norm3 = norm_layer(dim)
#         self.local_mp = LPI(in_features=dim, act_layer=act_layer)

#         # LayerScale Initialization
#         self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
#         self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
#         self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        
#         # Keypoint token
#         self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))

#     def forward(self, x, H, W):
#         if self.use_attn:
#             x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))

#         # LPI module processing
#         x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        
#         # Keypoint token concatenation if needed
#         if self.local_mp is None:
#             B, N, C = x.shape
#             keypoint_token = self.keypoint_token.expand(B, -1, -1)
#             x = torch.cat((x, keypoint_token), dim=1)
        
#         x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
#         return x




# class Block(nn.Module):
#     def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
#                  drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm, attn_head_dim=None, eta=1.0, use_attn=True, num_keypoints=10):
#         super().__init__()
#         self.use_attn = use_attn
#         self.num_keypoints = num_keypoints

#         self.norm1 = norm_layer(dim)
#         if use_attn:
#             self.attn = Attention1(
#                 dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
#             )

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         self.norm3 = norm_layer(dim)
#         self.local_mp = LPI(in_features=dim, act_layer=act_layer)

#         # LayerScale Initialization
#         self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
#         self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
#         self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        
#         # Keypoint token
#         self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))

#     def forward(self, x, H, W):
#         if self.use_attn:
#             x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x),H,W))

#         if self.local_mp is not None:
#             x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
#         else:
#             B, N, C = x.shape
#             keypoint_token = self.keypoint_token.expand(B, -1, -1)
#             x = torch.cat((x, keypoint_token), dim=1)

#         x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
#         return x
    
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None, eta=1.0, use_attn=True, num_keypoints=10):
        super().__init__()
        self.use_attn = use_attn
        self.num_keypoints = num_keypoints

        self.norm1 = norm_layer(dim)
        if use_attn:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer)

        # LayerScale Initialization
        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        
        # Keypoint token
        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))

    def forward(self, x, H, W):
        if self.use_attn:
            x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x),H,W))

        # x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        

        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x





class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding=4 + 2 * (ratio//2-1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)
    


def tuple_div(tp1, tp2):
    return tuple(i // j for i, j in zip(tp1, tp2))



class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x




@BACKBONES.register_module()
class ViT(BaseBackbone):
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,
                 num_keypoints=16):
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth
        self.num_keypoints = num_keypoints
        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, 768))
        
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches

        # since the pretraining model has class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_attn=True,
                num_keypoints=self.num_keypoints
                )
            for i in range(depth)])
        

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def forward_features(self, x):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x,Hp,Wp)
            else:
                x = blk(x,Hp,Wp)

        x = self.last_norm(x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        return xp


    

  
    
    
    
# @BACKBONES.register_module()
# class ViT(BaseBackbone):
#     def __init__(self,
#                  img_size=(256, 192), patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
#                  num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
#                  drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
#                  frozen_stages=-1, ratio=1, last_norm=True,
#                  patch_padding='pad', freeze_attn=False, freeze_ffn=False,
#                  alpha=0.5, num_keypoints=16):
#         super(ViT, self).__init__()
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim
#         self.frozen_stages = frozen_stages
#         self.use_checkpoint = use_checkpoint
#         self.patch_padding = patch_padding
#         self.freeze_attn = freeze_attn
#         self.freeze_ffn = freeze_ffn
#         self.depth = depth
#         self.alpha = alpha
#         self.num_keypoints = num_keypoints
#         self.img_size = img_size  # Ensure img_size is correctly defined as a tuple

#         if hybrid_backbone is not None:
#             self.patch_embed = HybridEmbed(
#                 hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
#         else:
#             self.patch_embed = PatchEmbed(
#                 img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio)
#         self.num_patches_list = self.patch_embed.num_patches_list

#         self.pos_embed_list = nn.ParameterList([
#             nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
#             for num_patches in self.num_patches_list
#         ])
#         self.pos_drop = nn.Dropout(p=drop_rate)

#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

#         self.blocks = nn.ModuleList([
#             Block(
#                 dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
#                 )
#             for i in range(depth)])

#         self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

#         for pos_embed in self.pos_embed_list:
#             trunc_normal_(pos_embed, std=.02)

#         self._freeze_stages()

#         # Define keypoint tokens and quality tokens
#         self.keypoint_tokens = nn.Parameter(torch.zeros(1, num_keypoints, embed_dim))
#         self.quality_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

#         # Quality predictor
#         self.quality_predictor = QualityPredictor(embed_dim, drop=drop_rate, sigmoid=False, qp_abs=False)

#         # Reuse block for coarse stage features
#         self.reuse_block = nn.Sequential(
#             norm_layer(embed_dim),
#             Mlp(in_features=embed_dim, hidden_features=mlp_ratio * embed_dim, out_features=embed_dim, act_layer=nn.GELU, drop=drop_rate)
#         )
        

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            logger.info(f"load from {pretrained}")
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

#     def forward_features(self, img):
#         results = []
#         global_attention = 0

#         # Coarse stage
#         x = F.interpolate(img, size=self.img_size[0], mode="bilinear")
#         B = x.shape[0]
#         x, _ = self.patch_embed(x)
#         x = x + self.pos_embed_list[0]
#         keypoint_tokens = self.keypoint_tokens.expand(B, -1, -1)
#         quality_tokens = self.quality_token.expand(B, -1, -1)
#         x = torch.cat((quality_tokens, keypoint_tokens, x), dim=1)
#         x = self.pos_drop(x)
#         embedding_x1 = x
#         for index, blk in enumerate(self.blocks):
#             x, atten = blk(x)
#             if index in self.target_index:
#                 global_attention = self.beta * global_attention + (1 - self.beta) * atten
#         x = self.norm(x)
#         self.global_attention = global_attention
#         quality_tokens, keypoint_tokens, feature_temp = torch.split_with_sizes(x,
#                                                                                [1, self.num_keypoints,
#                                                                                 self.patch_embed.num_patches_list[0]], dim=1)
#         results.append(keypoint_tokens)

#         # Fine stage
#         x = F.interpolate(img, size=self.img_size[1], mode="bilinear")
#         x, _ = self.patch_embed(x, img_size_idx=1)
#         x = x + self.pos_embed_list[1]
#         x = torch.cat((keypoint_tokens, x), dim=1)

#         embedding_x2 = x + feature_temp  # Shortcut
#         if self.informative_selection:
#             keypoints_attn = global_attention.mean(dim=1)[:, 1:self.num_keypoints, self.num_keypoints + 1:].sum(
#                 dim=1)
#             import_token_num = math.ceil(self.alpha * self.patch_embed.num_patches_list[0])
#             policy_index = torch.argsort(keypoints_attn, dim=1, descending=True)
#             unimportant_index = policy_index[:, import_token_num:]
#             important_index = policy_index[:, :import_token_num]
#             unimportant_tokens = batch_index_select(embedding_x1, unimportant_index + self.num_keypoints + 1)
#             important_index = get_index(important_index,
#                                         patch_shape_src=self.patch_embed.patch_shapes[0],
#                                         patch_shape_des=self.patch_embed.patch_shapes[1])
#             cls_index = torch.arange(self.num_keypoints, device=x.device).unsqueeze(0).repeat(B, 1)
#             important_index = torch.cat((cls_index, important_index + self.num_keypoints), dim=1)
#             important_tokens = batch_index_select(embedding_x2, important_index)
#             x = torch.cat((important_tokens, unimportant_tokens), dim=1)

#         if self.replace_oks:
#             quality_tokens = self.quality_token.expand(B, -1, -1)
#             x = torch.cat((quality_tokens, x), dim=1)
#         x = self.pos_drop(x)
#         for blk in self.blocks:
#             x, _ = blk(x)
#         x = self.norm(x)
#         if self.replace_oks:
#             quality_tokens = x[:, :1]
#             quality_fine = self.quality_predictor(quality_tokens)
#             keypoint_tokens = x[:, 1:self.num_keypoints + 1]
#         else:
#             keypoint_tokens = x[:, :self.num_keypoints]

#         results.append(keypoint_tokens)

#         return results


    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        super().init_weights(pretrained, patch_padding=self.patch_padding)

        if pretrained is None:
            def _init_weights(m):
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

            self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}



    def forward(self, x):
        x = self.forward_features(x)
        return x

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
