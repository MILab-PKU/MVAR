# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
import math
import logging
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as torch_checkpoint
from timm.models.layers import PatchEmbed

from utils.drop_path import DropPath
from transformers import AutoModel


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    num_viewes: int = 8
    n_head: int = 32
    n_kv_head: Optional[int] = None
    kv_ratio: int = 2
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    lora_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    camera_dim: int = 1
    class_dropout_prob: float = 0.1
    model_type: str = 't&cam2i'
    adapter_size: str = 'small'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        # T = caption.shape[1]
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        # drop_ids = drop_ids.unsqueeze(1).repeat(1, T)
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class ShapeEmbedder(nn.Module):
    """
    Embeds shape caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.uncond_prob = uncond_prob

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        embeddings = self.cap_proj(caption)
        if (train and use_dropout) or (force_drop_ids is not None):
            embeddings = torch.zeros_like(embeddings)
        return embeddings

#################################################################################
#                      Embedding Layers for Camera Pose                         #
#################################################################################
class CameraEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=16):
        super().__init__()
        self.cam_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)

    def forward(self, cam_pose):
        embeddings = self.cam_proj(cam_pose.unsqueeze(-1))
        return embeddings


#################################################################################
#                      Embedding Layers for Depth Condition                     #
#################################################################################
class ConditionEmbedder(nn.Module):
    """
    Embeds Condition into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120, vocab_size=16384):
        super().__init__()
        self.cap_proj = MLP(in_features=hidden_size, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", torch.zeros(token_num, hidden_size) / hidden_size ** 0.5)
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None, drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            if drop_ids is None:
                drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1

        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding[:caption.shape[1]], caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None, drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids, drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))
    
    def reset(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.k_cache = torch.zeros(cache_shape, dtype=dtype)
        self.v_cache = torch.zeros(cache_shape, dtype=dtype)

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class MultiViewKVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_l_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('k_r_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_l_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_r_cache', torch.zeros(cache_shape, dtype=dtype))
    
    def reset(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.k_l_cache = torch.zeros(cache_shape, dtype=dtype)
        self.k_r_cache = torch.zeros(cache_shape, dtype=dtype)
        self.v_l_cache = torch.zeros(cache_shape, dtype=dtype)
        self.v_r_cache = torch.zeros(cache_shape, dtype=dtype)
    
    def update_cross_view(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        if input_pos.shape[0] > 1:
            assert input_pos.shape[0] == 137
            input_pos = torch.tensor([0], device=k_val.device).int()
        else:
            input_pos = input_pos - 136
        assert input_pos.shape[0] * 2 == k_val.shape[2] == 2
        k_l_out = self.k_l_cache
        k_r_out = self.k_r_cache
        v_l_out = self.v_l_cache
        v_r_out = self.v_r_cache
        k_l, k_r = k_val.chunk(2, dim=-2)
        v_l, v_r = v_val.chunk(2, dim=-2)
        k_l_out[:, :, input_pos] = k_l
        k_r_out[:, :, input_pos] = k_r
        v_l_out[:, :, input_pos] = v_l
        v_r_out[:, :, input_pos] = k_r

        return torch.cat([k_l_out, k_r_out], dim=-2), torch.cat([v_l_out, v_r_out], dim=-2)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # xq = apply_rotary_emb(xq, freqs_cis)
        # xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))
        # print(xk.shape, xv.shape)
        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        # print(keys.shape, values.shape)
        # print()
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class AttentionLoRA(nn.Module):
    def __init__(self, config: ModelArgs, rank: int, alpha: int = None):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim
        self.rank = rank
        self.alpha = alpha or rank

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None
        self.lora_q = nn.Sequential(
            nn.Linear(config.dim, rank, bias=False),
            nn.Linear(rank, config.dim, bias=False)
        )
        self.lora_v = nn.Sequential(
            nn.Linear(config.dim, rank, bias=False),
            nn.Linear(rank, config.dim, bias=False)
        )

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)
        self.lora_dropout_q = nn.Dropout(config.lora_dropout_p)
        self.lora_dropout_v = nn.Dropout(config.lora_dropout_p)

        self.wqkv.weight.requires_grad = False
        self.wo.weight.requires_grad = False
        self.lora_q.requires_grad_()
        self.lora_v.requires_grad_()

    def _init_lora(self):
        # gaussian init
        nn.init.normal_(self.lora_q[0].weight, std=1 / self.rank)
        nn.init.zeros_(self.lora_q[1].weight)
        nn.init.normal_(self.lora_v[0].weight, std=1 / self.rank)
        nn.init.zeros_(self.lora_v[1].weight)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)
        lora_q_out = self.lora_q(self.lora_dropout_q(x))
        lora_v_out = self.lora_v(self.lora_dropout_v(x))
        # RELora here: alpha / math.sqrt(rank), arXiv.2312.03732
        xq = xq + (self.alpha / math.sqrt(self.rank)) * lora_q_out
        xv = xv + (self.alpha / math.sqrt(self.rank)) * lora_v_out

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # xq = apply_rotary_emb(xq, freqs_cis)
        # xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
            assert mask is not None
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float, lora: bool):
        super().__init__()

        self.attention = Attention(config)
        if lora:
            del self.attention
            self.attention = AttentionLoRA(config, rank=4)

        self.feed_forward = FeedForward(config)

        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if lora:
            self.feed_forward.requires_grad_(False)
            self.ffn_norm.requires_grad_(False)

    def forward(
        self, x: torch.Tensor, T: int, num_image_token: int, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None, prefill: bool = False):
        B, N, C = x.shape

        # stage 1: self attention on spatial
        shortcut_x = x
        x_attn = self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        h = shortcut_x + x_attn

        # stage 2: cross-view attention on temporal-spatial use split-token linear attention
        if hasattr(self, "temporal_attention"):
            h_non_input, h_input = h.split([N - num_image_token, num_image_token], dim=-2)
            shortcut_h = h_input
            h_attn = self.drop_path(self.gamma * self.temporal_attention(self.temporal_attention_norm(h_input), input_pos=start_pos, mask=mask))
            h = shortcut_h + h_attn.reshape(B, T, num_image_token, C)
            h = torch.cat([h_non_input, h], dim=-2)

        # stage 3: ffn
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class SplitTransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float, lora: bool):
        super().__init__()

        self.attention = Attention(config)
        if lora:
            del self.attention
            self.attention = AttentionLoRA(config, rank=64)

        self.feed_forward = FeedForward(config)

        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if lora:
            self.feed_forward.requires_grad_(False)
            self.ffn_norm.requires_grad_(False)

    def forward(
        self, x: torch.Tensor, T: int, num_image_token: int, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None, prefill: bool = False):
        B, N, C = x.shape

        # stage 1: split token attention on spatial
        shortcut_x = x
        x_attn = self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        if self.training or prefill:
            # num_image_token = 120
            x_cond, x_image = x_attn.reshape(B, N, C).split([N - num_image_token, num_image_token], dim=-2)
            x_attn = torch.cat([torch.zeros_like(x_cond), x_image], dim=-2)
        else:
            x_attn = x_attn.reshape(B, N, C)
        h = shortcut_x + x_attn

        # stage 2: cross-view attention on temporal-spatial use split-token linear attention
        if hasattr(self, "temporal_attention"):
            h_non_input, h_input = h.split([N - num_image_token, num_image_token], dim=-2)
            shortcut_h = h_input
            h_attn = self.drop_path(self.temporal_attention(self.temporal_attention_norm(h_input), input_pos=start_pos, mask=mask))
            h = shortcut_h + h_attn.reshape(B, num_image_token, C)
            h = torch.cat([h_non_input, h], dim=-2)

        # stage 3: ffn
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNSplitTransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float, lora: bool):
        super().__init__()
        self.ada_linear = nn.Linear(config.dim, 6 * config.dim)

        self.attention = Attention(config)
        if lora:
            del self.attention
            self.attention = AttentionLoRA(config, rank=64)

        self.feed_forward = FeedForward(config)

        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if lora:
            self.feed_forward.requires_grad_(False)
            self.ffn_norm.requires_grad_(False)
    
    def _init_adaln(self):
        nn.init.zeros_(self.ada_linear.weight.data)
        nn.init.zeros_(self.ada_linear.bias.data)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, T: int, num_image_token: int, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None, prefill: bool = False):
        B, N, C = x.shape

        shift_msa, scale_msa, shift_mlp, scale_mlp, gate_msa, gate_mlp = self.ada_linear(cond).chunk(6, dim=1)

        # stage 1: split token attention on spatial        
        shortcut_x = x
        x_attn = self.drop_path(self.attention(
            modulate(self.attention_norm(x), shift_msa, scale_msa)
        , freqs_cis, start_pos, mask))
        if self.training or prefill:
            # num_image_token = 120
            x_cond, x_image = x_attn.reshape(B, N, C).split([N - num_image_token, num_image_token], dim=-2)
            x_attn = torch.cat([torch.zeros_like(x_cond), x_image], dim=-2)
        else:
            x_attn = x_attn.reshape(B, N, C)
        h = shortcut_x + (1 + gate_msa.unsqueeze(1)) * x_attn

        # stage 2: cross-view attention on temporal-spatial use split-token linear attention
        if hasattr(self, "temporal_attention"):
            h_non_input, h_input = h.split([N - num_image_token, num_image_token], dim=-2)
            shortcut_h = h_input
            h_attn = self.drop_path(self.temporal_attention(self.temporal_attention_norm(h_input), input_pos=start_pos, mask=mask))
            h = shortcut_h + h_attn.reshape(B, num_image_token, C)
            h = torch.cat([h_non_input, h], dim=-2)

        # stage 3: ffn
        out = h + (1 + gate_mlp.unsqueeze(1)) * self.drop_path(self.feed_forward(modulate(self.ffn_norm(h), shift_mlp, scale_mlp)))
        # out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Dinov2_Adapter(nn.Module):
    def __init__(self, adapter_size='base', condition_type='depth'):
        super(Dinov2_Adapter, self).__init__()
        assert condition_type == "depth"
        self.model = AutoModel.from_pretrained(f'autoregressive/models/dinov2-{adapter_size}')
        self.model.requires_grad_(False)
        self.condition_type = condition_type
    
    def to_patch14(self, input):
        input = input.flatten(0, 1)
        H, W = input.shape[2:]
        new_H = (H // 16) * 14
        new_W = (W // 16) * 14
        # if self.condition_type in ['canny', 'seg']:
        #     output = torch.nn.functional.interpolate(input, size=(new_H, new_W), mode='nearest')#, align_corners=True)  canny, seg
        # else:
        output = torch.nn.functional.interpolate(input, size=(new_H, new_W), mode='bicubic', align_corners=True) # depth, lineart, hed
        # TODO: case
        return output.bfloat16()

    @torch.no_grad()
    def forward(self, x):
        x = self.to_patch14(x)
        x = self.model(x)
        return x.last_hidden_state[:, 1:]


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_viewes = config.num_viewes
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.layer_internal = config.n_layer // 3

        # # Depth control adapter
        # self.adapter = Dinov2_Adapter(config.adapter_size)
        # if config.adapter_size == "small":
        #     self.adapter_mlp = MLP(384, config.dim, config.dim)
        # elif config.adapter_size == "base":
        #     self.adapter_mlp = MLP(768, config.dim, config.dim)

        self.tok_embeddings = nn.Embedding(config.vocab_size + 1, config.dim)
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        elif self.model_type == 't&cam2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
            # img_size, TODO: hard code
            self.cam_patchify = PatchEmbed(img_size=256, patch_size=16, in_chans=6, embed_dim=config.dim)
            self.camera_gamma = 1
        elif self.model_type == 'i&cam2i':
            # img_size, TODO: hard code
            self.cam_patchify = PatchEmbed(img_size=256, patch_size=16, in_chans=6, embed_dim=config.dim)
            self.camera_gamma = nn.Parameter(torch.zeros((1, 1, config.dim)), requires_grad=True)
        elif self.model_type == 'ti&cam2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
            # img_size, TODO: hard code
            self.cam_patchify = PatchEmbed(img_size=256, patch_size=16, in_chans=6, embed_dim=config.dim)
            self.camera_gamma = 1
        elif self.model_type == 't&shape_mv':
            shape_token_dim = 64 # TODO: hard code
            self.shape_proj = ShapeEmbedder(in_channels=shape_token_dim, hidden_size=config.dim, uncond_prob=config.class_dropout_prob)
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob+0.05)
            # img_size, TODO: hard code
            self.cam_patchify = PatchEmbed(img_size=256, patch_size=16, in_chans=6, embed_dim=config.dim)
            self.camera_gamma = 1
        else:
            raise Exception("please check model type")
        self.tok_dropout = nn.Dropout(config.token_dropout_p)
        
        # # condition layers
        # self.condition_layers = torch.nn.ModuleList()
        # for _ in range(3):
        #     self.condition_layers.append(MLP(config.dim, config.dim, config.dim))

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(AdaLNSplitTransformerBlock(config, dpr[layer_id], lora=False))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        if self.model_type == "t&cam2i":
            self.cls_token_num += 1  # no camera pose and EOS uses one.
        elif self.model_type == "ti&cam2i":
            # NOTE: 1025 for 512 resolution, 257 for 256 resolution
            self.cls_token_num += 256  # 16 (32) * 16 + 1, with one reference image
        elif self.model_type == "t&shape_mv":
            self.cls_token_num += 256 # 256 for shape token length
            self.cls_token_num += 2 # one for start of shape, one for start of image
        # self.cls_token_num += 17  # Camera pose uses 16 tokens and EOS uses one.
        self.freqs_cis = None

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1
        self.kvcache = False

        self.cond_ada = None

        self.use_checkpoint = True
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Initialize LoRA and adaLN
        for m in self.modules():
            if hasattr(m, '_init_lora'):
                if not isinstance(m, AttentionLoRA):
                    logging.warning(f'Trying to call `_init_lora()` of {m}, which is not `AttentionLoRA`.')
                m._init_lora()
            if hasattr(m, '_init_adaln'):
                if not isinstance(m, AdaLNSplitTransformerBlock):
                    logging.warning(f'Trying to call `_init_lora()` of {m}, which is not `AttentionLoRA`.')
                m._init_adaln()

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, PatchEmbed):
            module.proj.weight.data.zero_()
            if module.proj.bias is not None:
                module.proj.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def reset_caches(self, dtype):
        head_dim = self.config.dim // self.config.n_head
        for b in self.layers:
            b.attention.kv_cache.reset(self.max_batch_size, self.max_seq_length, self.config.n_head, head_dim, dtype)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.config.dim // self.config.n_head
        # self.max_seq_length_image = max_seq_length - 137
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)
        self.kvcache = True

        causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        # self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

    @staticmethod
    def shuffle(x, orders):
        batch_size, seq_len = x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        shuffled_x = x[batch_indices, orders]
        return shuffled_x

    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # reference image
        camera_idx: torch.Tensor,
        # shape_token: torch.Tensor,
        # depth: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        T: int = None,
        prefill: bool = False,
        orders: Optional[torch.Tensor] = None,
    ):
        if idx is not None and cond_idx is not None and camera_idx is not None: # training or naive inference
            if self.training:
                input_pos = None
            # for one token prediction
            B, num_image_token = idx.shape
            num_image_token = num_image_token + 1
            num_viewes = num_image_token // self.block_size

            IMAGE_OF_START = self.vocab_size * torch.ones((B, 1)).type_as(idx)

            if self.model_type == 'i&cam2i':
                # condition: reference image
                cond_embeddings = self.tok_embeddings(cond_idx)
            elif self.model_type == 't&cam2i':
                # condition: text
                cond_embeddings = self.cls_embedding(cond_idx, train=self.training)
            elif self.model_type == 'ti&cam2i':
                # cond_idx[0]: reference image
                # cond_idx[1]: text
                image_cond_embeddings = self.tok_embeddings(cond_idx[0]).flatten(1, 2)
                text_cond_embeddings = self.cls_embedding(cond_idx[1], train=self.training)
                cond_embeddings = torch.cat([text_cond_embeddings, image_cond_embeddings], dim=1)
            elif self.model_type == 't&shape_mv':
                # cond_idx[0]: shape token
                # cond_idx[1]: text
                shape_cond_embeddings = self.shape_proj(cond_idx[0], train=self.training).flatten(1, 2)
                text_cond_embeddings = self.cls_embedding(cond_idx[1], train=self.training)

                SHAPE_OF_START = self.vocab_size * torch.zeros((B, 1)).type_as(idx)
                token_embeddings_shape = self.tok_embeddings(SHAPE_OF_START)

                cond_embeddings = torch.cat([text_cond_embeddings, token_embeddings_shape, shape_cond_embeddings], dim=1)
                
            cond_embeddings = cond_embeddings[:, :self.cls_token_num] # B, N, C
            cond_ada = cond_embeddings.mean(dim=-2)

            # image token, concat with camera pose
            if self.model_type == 'i&cam2i' or self.model_type == 'ti&cam2i':
                # for reference image condition:
                camera_embeddings = self.cam_patchify(camera_idx[:, 1:, ...].flatten(0, 1)) # TODO: for only one reference image
            elif self.model_type == 't&cam2i':
                # for text condition:
                camera_embeddings = self.cam_patchify(camera_idx.flatten(0, 1))
            else:
                camera_embeddings = self.cam_patchify(camera_idx.flatten(0, 1))

            camera_embeddings = camera_embeddings * self.camera_gamma

            # for one token prediction
            camera_embeddings = camera_embeddings.reshape(B, num_viewes, -1, self.dim).flatten(1, 2)
            camera_embeddings = self.shuffle(camera_embeddings, orders)

            # for one token prediction
            token_embeddings = self.tok_embeddings(torch.cat([IMAGE_OF_START, idx], dim=-1)) + camera_embeddings

            # concat
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=-2)
            h = self.tok_dropout(token_embeddings)
        else:

            if prefill: # prefill in inference
                # condition embeddings
                if self.model_type == 'i&cam2i':
                    cond_embeddings = self.tok_embeddings(cond_idx)[:,:self.cls_token_num]
                elif self.model_type == 't&cam2i':
                    cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                elif self.model_type == 'ti&cam2i':
                    image_cond_embeddings = cond_idx[0]
                    text_cond_embeddings = self.cls_embedding(cond_idx[1], train=self.training)
                    cond_embeddings = torch.cat([text_cond_embeddings, image_cond_embeddings], dim=1)[:,:self.cls_token_num]
                elif self.model_type == 't&shape_mv':
                    # cond_idx[0]: shape token
                    # cond_idx[1]: text
                    shape_cond_embeddings = self.shape_proj(cond_idx[0], train=self.training)
                    text_cond_embeddings = self.cls_embedding(cond_idx[1], train=self.training)
                    cond_embeddings = torch.cat([text_cond_embeddings, shape_cond_embeddings], dim=1)
                    
                cond_embeddings = cond_embeddings[:, :self.cls_token_num] # B, N, C
                self.cond_ada = cond_ada = cond_embeddings.mean(dim=-2)
                # token embeddings + rays embeddings
                SHAPE_OF_START = self.vocab_size * torch.ones((cond_embeddings.shape[0], 1)).to(cond_embeddings.device).long()
                # token_embeddings = torch.cat([self.tok_embeddings(IMGAE_OF_START), camera_idx[:, 0:1]], dim=-1)
                # token_embeddings = self.condition_mlp(token_embeddings)
                token_embeddings = self.tok_embeddings(SHAPE_OF_START) + camera_idx[:, 0:1]

                # concat
                token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=-2)
                num_image_token = 1
            else: # decode_n_tokens (kv cache) in inference
                cond_ada = self.cond_ada

                token_embeddings = self.tok_embeddings(idx) + camera_idx[:, input_pos - self.cls_token_num + 1]

                num_image_token = 1

            if mask is None:
                mask = self.causal_mask[:, None, input_pos]

            h = token_embeddings

        freqs_cis = None

        # transformer blocks
        for i, layer in enumerate(self.layers):
            # checkpoint
            if self.use_checkpoint:
                h = torch_checkpoint.checkpoint(layer, h, cond_ada, T, num_image_token, freqs_cis, input_pos, mask, prefill)
            else:
                h = layer(h, cond_ada, T, num_image_token, freqs_cis, input_pos, mask, prefill)

        # output layers
        h = self.norm(h)
        loss, loss_ref = None, None
        if self.training and self.model_type == "ti&cam2i":
            h_ref = h[:, 120:-num_image_token].contiguous()
            loss_ref = F.l1_loss(h_ref, image_cond_embeddings, reduction="mean")

        logits = self.output(h).float()
        if self.training:
            logits = logits[:, -num_image_token:].contiguous()

        # if we are given some desired targets also calculate the loss
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[-1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, (loss, loss_ref)

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)


#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def NVCGPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def NVCGPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def NVCGPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def NVCGPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def NVCGPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def NVCGPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def NVCGPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def NVCGPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M


NVCGPT_models = {
    'NVCGPT-B': NVCGPT_B, 'NVCGPT-L': NVCGPT_L, 'NVCGPT-XL': NVCGPT_XL, 'NVCGPT-XXL': NVCGPT_XXL, 'NVCGPT-XXXL': NVCGPT_XXXL,
    'NVCGPT-1B': NVCGPT_1B, 'NVCGPT-3B': NVCGPT_3B, 'NVCGPT-7B': NVCGPT_7B, 
}