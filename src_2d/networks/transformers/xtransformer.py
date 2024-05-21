import math
from functools import partial
from typing import Optional, Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum
from x_transformers.x_transformers import (
    TokenEmbedding,
    AbsolutePositionalEmbedding,
    default,
    always,
    exists,
    groupby_prefix_and_trim,
    FixedPositionalEmbedding,
    RotaryEmbedding,
    DynamicPositionBias,
    LearnedAlibiPositionalBias,
    AlibiPositionalBias,
    RelativePositionBias,
    ScaleNorm,
    RMSNorm,
    Rezero,
    Scale,
    not_equals,
    equals,
    cast_tuple,
    ShiftTokens,
    FeedForward,
    GRUGating,
    Residual,
    LayerIntermediates,
    Attention,
    apply_rotary_pos_emb,
    stable_softmax,
    init_zero_,
    l2norm,
    max_neg_value
)
from entmax import entmax15


from src.networks.transformers.img2seq_ordering import (
    Ordering,
    RelativeSpatialPositioning,
)
from src.networks.transformers.transformer import TransformerBase
from src.utils.constants import (
    TransformerConditioningType,
    TransformerSpatialConditioningType,
)
from src.utils.transformer import (
    AbsoluteSpatialPositionalEmbedding,
    FixedSpatialPositionalEmbedding,
)
from collections import namedtuple


DEFAULT_DIM_HEAD = 64


Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])


class SpatialRelativePositionBias(nn.Module):
    def __init__(self, scale, spatial_dist_matrix, num_buckets=32, heads=8):
        super().__init__()
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)
        self.scale = scale
        self.register_buffer(name="rp_buckets", tensor=spatial_dist_matrix)

    def forward(self, qk_dots):
        rp_buckets = self.rp_buckets[: qk_dots.shape[2], : qk_dots.shape[3]]
        values = self.relative_attention_bias(rp_buckets)
        bias = rearrange(values, "i j h -> () h i j")

        return qk_dots + (bias + self.scale)


class DoubleAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = DEFAULT_DIM_HEAD,
        heads = 8,
        causal = False,
        talking_heads = False,
        head_scale = False,
        sparse_topk = None,
        use_entmax15 = False,
        num_mem_kv = 0,
        dropout = 0.,
        on_attn = False,
        gate_values = False,
        zero_init_output = False,
        max_attend_past = None,
        qk_norm = False,
        scale_init_value = None,
        one_kv_head = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.causal = causal
        self.max_attend_past = max_attend_past

        q_dim = k_dim = v_dim = out_dim = dim_head * heads

        self.one_kv_head = one_kv_head
        if one_kv_head:
            k_dim = v_dim = dim_head
            out_dim = v_dim * heads

        self.to_q = nn.Linear(dim, q_dim, bias = False)
        self.to_k = nn.Linear(dim, k_dim, bias = False)
        self.to_v = nn.Linear(dim, v_dim, bias = False)

        self.to_q_2 = nn.Linear(dim, q_dim, bias = False)
        self.to_k_2 = nn.Linear(dim, k_dim, bias = False)
        self.to_v_2 = nn.Linear(dim, v_dim, bias = False)

        self.dropout = nn.Dropout(dropout)


        # add GLU gating for aggregated values, from alphafold2
        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, out_dim)
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 1)
            self.to_v_gate_2 = nn.Linear(dim, out_dim)
            nn.init.constant_(self.to_v_gate_2.weight, 0)
            nn.init.constant_(self.to_v_gate_2.bias, 1)

        # cosine sim attention
        self.qk_norm = qk_norm
        if qk_norm:
            scale_init_value = default(scale_init_value, -3) # if not provided, initialize as though it were sequence length of 1024
            self.scale = nn.Parameter(torch.ones(1, heads, 1, 1) * scale_init_value)
            self.scale_2 = nn.Parameter(torch.ones(1, heads, 1, 1) * scale_init_value)

        # talking heads
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
            self.post_softmax_talking_heads = nn.Conv2d(heads, heads, 1, bias = False)
            self.pre_softmax_talking_heads_2 = nn.Conv2d(heads, heads, 1, bias = False)
            self.post_softmax_talking_heads_2 = nn.Conv2d(heads, heads, 1, bias = False)

        # head scaling
        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))
            self.head_scale_params_2 = nn.Parameter(torch.ones(1, heads, 1, 1))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # entmax
        self.attn_fn = entmax15 if use_entmax15 else stable_softmax

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_k_2 = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v_2 = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(nn.Linear(out_dim, dim * 2, bias = False), nn.GLU()) if on_attn else nn.Linear(out_dim, dim, bias = False)
        self.to_out_2 = nn.Sequential(nn.Linear(out_dim, dim * 2, bias = False), nn.GLU()) if on_attn else nn.Linear(out_dim, dim, bias = False)

        # init output projection 0
        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        attn_mask = None,
        rel_pos = None,
        sinusoidal_emb = None,
        rotary_pos_emb = None,
        prev_attn = None,
        mem = None
    ):
        num_layers = 2
        for l in range(num_layers):
            b, n, _, h, talking_heads, head_scale, scale, device, has_context = *x.shape, self.heads, self.talking_heads, self.head_scale, self.scale, x.device, exists(
                context)
            assert has_context, 'No Context Found for Double Cross Attention Layer'

            kv_input = context[l]

            q_input = x
            k_input = kv_input
            v_input = kv_input

            if exists(mem):
                k_input = torch.cat((mem, k_input), dim = -2)
                v_input = torch.cat((mem, v_input), dim = -2)

            if exists(sinusoidal_emb):
                # in shortformer, the query would start at a position offset depending on the past cached memory
                offset = k_input.shape[-2] - q_input.shape[-2]
                q_input = q_input + sinusoidal_emb(q_input, offset = offset)
                k_input = k_input + sinusoidal_emb(k_input)

            if l == 0:
                q, k, v = self.to_q(q_input), self.to_k(k_input), self.to_v(v_input)
            else:
                q, k, v = self.to_q_2(q_input), self.to_k_2(k_input), self.to_v_2(v_input)

            q = rearrange(q, 'b n (h d) -> b h n d', h = h)

            if not self.one_kv_head:
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (k, v))

            if exists(rotary_pos_emb) and not has_context:
                l = rotary_pos_emb.shape[-1]
                (ql, qr), (kl, kr), (vl, vr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k, v))
                ql, kl, vl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl, vl))
                q, k, v = map(lambda t: torch.cat(t, dim = -1), ((ql, qr), (kl, kr), (vl, vr)))

            input_mask = None
            if any(map(exists, (mask, context_mask))):
                q_mask = default(mask, lambda: torch.ones((b, n), device = device).bool())
                k_mask = q_mask if not exists(context) else context_mask
                k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device = device).bool())
                q_mask = rearrange(q_mask, 'b i -> b () i ()')
                k_mask = rearrange(k_mask, 'b j -> b () () j')
                input_mask = q_mask * k_mask

            if self.num_mem_kv > 0:
                if l == 0:
                    mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), (self.mem_k, self.mem_v))
                else:
                    mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), (self.mem_k_2, self.mem_v_2))
                k = torch.cat((mem_k, k), dim = -2)
                v = torch.cat((mem_v, v), dim = -2)
                if exists(input_mask):
                    input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value = True)

            if self.qk_norm:
                q, k = map(l2norm, (q, k))
                if l == 0:
                    scale = 1 / (self.scale.exp().clamp(min = 1e-2))
                else:
                    scale = 1 / (self.scale_2.exp().clamp(min = 1e-2))

            kv_einsum_eq = 'b h j d' if not self.one_kv_head else 'b j d'

            dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale
            mask_value = max_neg_value(dots)

            if exists(prev_attn):
                dots = dots + prev_attn

            pre_softmax_attn = dots.clone()

            if talking_heads:
                if l == 0:
                    dots = self.pre_softmax_talking_heads(dots)
                else:
                    dots = self.pre_softmax_talking_heads_2(dots)

            if exists(rel_pos):
                if l == 0:
                    dots = rel_pos(dots)

            if exists(input_mask):
                dots.masked_fill_(~input_mask, mask_value)
                del input_mask

            if exists(attn_mask):
                assert 2 <= attn_mask.ndim <= 4, 'attention mask must have greater than 2 dimensions but less than or equal to 4'
                if attn_mask.ndim == 2:
                    attn_mask = rearrange(attn_mask, 'i j -> () () i j')
                elif attn_mask.ndim == 3:
                    attn_mask = rearrange(attn_mask, 'h i j -> () h i j')
                dots.masked_fill_(~attn_mask, mask_value)

            if exists(self.max_attend_past):
                i, j = dots.shape[-2:]
                range_q = torch.arange(j - i, j, device = device)
                range_k = torch.arange(j, device = device)
                dist = rearrange(range_q, 'i -> () () i ()') - rearrange(range_k, 'j -> () () () j')
                mask = dist > self.max_attend_past
                dots.masked_fill_(mask, mask_value)
                del mask

            if self.causal:
                i, j = dots.shape[-2:]
                r = torch.arange(i, device = device)
                mask = rearrange(r, 'i -> () () i ()') < rearrange(r, 'j -> () () () j')
                mask = F.pad(mask, (j - i, 0), value = False)
                dots.masked_fill_(mask, mask_value)
                del mask

            if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
                top, _ = dots.topk(self.sparse_topk, dim = -1)
                vk = top[..., -1].unsqueeze(-1).expand_as(dots)
                mask = dots < vk
                dots.masked_fill_(mask, mask_value)
                del mask

            attn = self.attn_fn(dots, dim = -1)
            post_softmax_attn = attn.clone()

            attn = self.dropout(attn)

            if talking_heads:
                if l == 0:
                    attn = self.post_softmax_talking_heads(attn)
                else:
                    attn = self.post_softmax_talking_heads_2(attn)

            out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

            if head_scale:
                if l == 0:
                    out = out * self.head_scale_params
                else:
                    out = out * self.head_scale_params_2

            out = rearrange(out, 'b h n d -> b n (h d)')

            if exists(self.to_v_gate):
                if l == 0:
                    gates = self.to_v_gate(x)
                    out = out * gates.sigmoid()
                else:
                    gates = self.to_v_gate_2(x)
                    out = out * gates.sigmoid()

            intermediates = Intermediates(
                pre_softmax_attn = pre_softmax_attn,
                post_softmax_attn = post_softmax_attn
            )

            if l == 0:
                out = self.to_out(out)
            else:
                out = self.to_out_2(out)

        return out, intermediates


class AttentionLayers(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads=8,
        causal=False,
        cross_attend=False,
        only_cross=False,
        use_scalenorm=False,
        use_rmsnorm=False,
        use_rezero=False,
        alibi_pos_bias=False,
        alibi_num_heads=None,
        alibi_learned=False,
        rel_pos_bias=False,
        spatial_rel_pos_bias=False,
        rel_pos_num_buckets=32,
        rel_pos_max_distance=128,
        dynamic_pos_bias=False,
        dynamic_pos_bias_log_distance=False,
        dynamic_pos_bias_mlp_depth=2,
        dynamic_pos_bias_norm=False,
        position_infused_attn=False,
        rotary_pos_emb=False,
        rotary_emb_dim=None,
        custom_layers=None,
        sandwich_coef=None,
        par_ratio=None,
        residual_attn=False,
        cross_residual_attn=False,
        macaron=False,
        pre_norm=True,
        gate_residual=False,
        scale_residual=False,
        scale_residual_constant=1.0,
        shift_tokens=0,
        sandwich_norm=False,
        use_qk_norm_attn=False,
        qk_norm_attn_seq_len=None,
        zero_init_branch_output=False,
        relative_spatial_pos_attr=None,
        double_cross_attend=False,
        **kwargs,
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim("ff_", kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim("attn_", kwargs)

        dim_head = attn_kwargs.get("dim_head", DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        self.has_pos_emb = position_infused_attn or rel_pos_bias or rotary_pos_emb
        self.pia_pos_emb = (
            FixedPositionalEmbedding(dim) if position_infused_attn else None
        )

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)
        self.rotary_pos_emb = (
            RotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None
        )

        assert not (
            alibi_pos_bias and rel_pos_bias
        ), "you can only choose Alibi positional bias or T5 relative positional bias, not both"
        assert (
            rel_pos_num_buckets <= rel_pos_max_distance
        ), "number of relative position buckets must be less than the relative position max distance"

        # relative positional bias
        self.rel_pos = None
        if rel_pos_bias:
            self.rel_pos = RelativePositionBias(
                scale=dim_head ** 0.5,
                causal=causal,
                heads=heads,
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
            )
        elif dynamic_pos_bias:
            self.rel_pos = DynamicPositionBias(
                dim=dim // 4,
                heads=heads,
                log_distance=dynamic_pos_bias_log_distance,
                depth=dynamic_pos_bias_mlp_depth,
                norm=dynamic_pos_bias_norm,
            )
        elif alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert (
                alibi_num_heads <= heads
            ), "number of ALiBi heads must be less than the total number of heads"
            alibi_pos_klass = (
                LearnedAlibiPositionalBias
                if alibi_learned or not causal
                else AlibiPositionalBias
            )
            self.rel_pos = alibi_pos_klass(
                heads=alibi_num_heads, bidirectional=not causal
            )
        elif spatial_rel_pos_bias:
            assert (
                relative_spatial_pos_attr is not None
            ), "Must have input Relative Spatial Positioning Attributes"
            self.rel_pos = SpatialRelativePositionBias(
                scale=dim_head ** 0.5,
                spatial_dist_matrix=relative_spatial_pos_attr.get_pid_array(),
                num_buckets=relative_spatial_pos_attr.get_num_pids() + 1,
                heads=heads,
            )

        assert not (
            not pre_norm and sandwich_norm
        ), "sandwich norm cannot be used when not using prenorm"
        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        self.cross_attend = cross_attend

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_fn = partial(norm_class, dim)

        norm_fn = nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None

        if double_cross_attend:
            default_block = ("a", "dc", "f")
        elif cross_attend and not only_cross:
            default_block = ("a", "c", "f")
        elif cross_attend and only_cross:
            default_block = ("c", "f")
        else:
            default_block = ("a", "f")

        if macaron:
            default_block = ("f",) + default_block

        # qk normalization

        if use_qk_norm_attn:
            attn_scale_init_value = (
                -math.log(math.log2(qk_norm_attn_seq_len ** 2 - qk_norm_attn_seq_len))
                if exists(qk_norm_attn_seq_len)
                else None
            )
            attn_kwargs = {
                **attn_kwargs,
                "qk_norm": True,
                "scale_init_value": attn_scale_init_value,
            }

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, "zero_init_output": True}
            ff_kwargs = {**ff_kwargs, "zero_init_output": True}

        # calculate layer block order

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, "par ratio out of range"
            default_block = tuple(filter(not_equals("f"), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = (
                par_depth * 2 // 3
            )  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert (
                len(default_block) <= par_width
            ), "default block is too large for par_ratio"
            par_block = default_block + ("f",) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ("f",) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert (
                sandwich_coef > 0 and sandwich_coef <= depth
            ), "sandwich coefficient should be less than the depth"
            layer_types = (
                ("a",) * sandwich_coef
                + default_block * (depth - sandwich_coef)
                + ("f",) * sandwich_coef
            )
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals("a"), layer_types)))

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(
            zip(self.layer_types, shift_tokens)
        ):
            is_last_layer = ind == (len(self.layer_types) - 1)

            if layer_type == "a":
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == "c":
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == "dc":
                layer = DoubleAttention(dim, heads=heads, **attn_kwargs)
            elif layer_type == "f":
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f"invalid layer type {layer_type}")

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if exists(branch_fn):
                layer = branch_fn(layer)

            if gate_residual:
                residual = GRUGating(dim, scale_residual=scale_residual)
            else:
                residual = Residual(
                    dim,
                    scale_residual=scale_residual,
                    scale_residual_constant=scale_residual_constant,
                )

            layer_uses_qk_norm = use_qk_norm_attn and layer_type in ("a", "c", "dc")

            pre_branch_norm = norm_fn() if pre_norm and not layer_uses_qk_norm else None
            post_branch_norm = (
                norm_fn() if sandwich_norm or layer_uses_qk_norm else None
            )
            post_main_norm = norm_fn() if not pre_norm and not is_last_layer else None

            norms = nn.ModuleList([pre_branch_norm, post_branch_norm, post_main_norm])

            self.layers.append(nn.ModuleList([norms, layer, residual]))

    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        mems=None,
        return_hiddens=False,
    ):
        assert not (
            self.cross_attend ^ exists(context)
        ), "context must be passed in if cross_attend is set to True"

        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            max_rotary_emb_length = max(
                list(map(lambda m: (m.shape[1] if exists(m) else 0) + x.shape[1], mems))
            )
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length, x.device)

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(
            zip(self.layer_types, self.layers)
        ):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == "a":
                hiddens.append(x)
                layer_mem = mems.pop(0) if mems else None

            residual = x

            pre_branch_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_branch_norm):
                x = pre_branch_norm(x)

            if layer_type == "a":
                out, inter = block(
                    x,
                    mask=mask,
                    attn_mask=attn_mask,
                    sinusoidal_emb=self.pia_pos_emb,
                    rel_pos=self.rel_pos,
                    rotary_pos_emb=rotary_pos_emb,
                    prev_attn=prev_attn,
                    mem=layer_mem,
                )
            elif layer_type == "c":
                out, inter = block(
                    x,
                    context=context,
                    mask=mask,
                    context_mask=context_mask,
                    prev_attn=prev_cross_attn,
                )
            elif layer_type == "f":
                out = block(x)

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, residual)

            if layer_type in ("a", "c"):
                intermediates.append(inter)

            if layer_type == "a" and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == "c" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn
            elif layer_type == "dc" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens, attn_intermediates=intermediates
            )

            return x, intermediates

        return x


class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on decoder"
        super().__init__(causal=True, **kwargs)


class XTransformer(TransformerBase):
    def __init__(
        self,
        dim,
        depth,
        num_tokens,
        max_seq_len,
        ordering: Ordering,
        emb_dim: int = None,
        max_mem_len: float = 0.0,
        shift_mem_down: int = 0,
        emb_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        num_memory_tokens: int = None,
        tie_embedding: bool = False,
        use_pos_emb: bool = True,
        l2norm_embed: bool = False,
        heads: int = 8,
        cross_attend: bool = False,
        only_cross: bool = False,
        use_scalenorm: bool = False,
        use_rmsnorm: bool = False,
        use_rezero: bool = False,
        alibi_pos_bias: bool = False,
        alibi_num_heads: int = None,
        alibi_learned: bool = False,
        rel_pos_bias: bool = False,
        rel_pos_num_buckets: int = 32,
        rel_pos_max_distance: int = 128,
        position_infused_attn: bool = False,
        rotary_pos_emb: bool = False,
        rotary_emb_dim: int = None,
        spatial_rel_pos_bias: bool = False,
        relative_spatial_pos_attr: RelativeSpatialPositioning = None,
        custom_layers=None,
        sandwich_coef: int = None,
        par_ratio: float = None,
        residual_attn: bool = False,
        cross_residual_attn: bool = False,
        macaron: bool = False,
        pre_norm: bool = True,
        gate_residual: bool = False,
        scale_residual: bool = False,
        shift_tokens: int = 0,
        sandwich_norm: bool = False,
        use_qk_norm_attn: bool = False,
        zero_init_branch_output: bool = False,
        ff_glu: bool = False,
        attn_talking_heads: bool = False,
        attn_on_attn: bool = False,
        attn_gate_values: bool = False,
        spatial_position_emb: str = None,
        spatial_shape: Union[Tuple[int, int], Tuple[int, int, int]] = None,
        conditioning_num_tokens: Optional[Tuple[int, ...]] = None,
        conditioning_type: str = TransformerConditioningType.NONE.value,
        num_tokens_enc: int = 0,
    ):
        super().__init__()

        self.conditioning_emb = nn.ModuleList()
        self.conditioning_type = conditioning_type
        self.cross_attend = (
            self.conditioning_type == TransformerConditioningType.CROSSATTEND.value
            or cross_attend
        )

        if num_tokens_enc != 0:
            self.conditioning_encoded_emb = nn.Embedding(num_tokens_enc, dim)
            self.cross_attend=True

        if (num_tokens_enc!=0) and (self.conditioning_type == TransformerConditioningType.CROSSATTEND.value) and conditioning_num_tokens:
            self.double_cross_attend = True
        else:
            self.double_cross_attend = False

        if conditioning_num_tokens:
            for count in conditioning_num_tokens:
                if count == -1:
                    self.conditioning_emb.append(None)
                else:
                    self.conditioning_emb.append(nn.Embedding(count, dim))

        attn_layers = Decoder(
            dim=dim,
            depth=depth,
            heads=heads,
            cross_attend=self.cross_attend,
            only_cross=only_cross,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
            use_scalenorm=use_scalenorm,
            use_rmsnorm=use_rmsnorm,
            use_rezero=use_rezero,
            alibi_pos_bias=alibi_pos_bias,
            alibi_num_heads=alibi_num_heads,
            alibi_learned=alibi_learned,
            rel_pos_bias=rel_pos_bias,
            spatial_rel_pos_bias=spatial_rel_pos_bias,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
            ff_glu=ff_glu,
            position_infused_attn=position_infused_attn,
            rotary_pos_emb=rotary_pos_emb,
            rotary_emb_dim=rotary_emb_dim,
            custom_layers=custom_layers,
            sandwich_coef=sandwich_coef,
            par_ratio=par_ratio,
            residual_attn=residual_attn,
            cross_residual_attn=cross_residual_attn,
            macaron=macaron,
            pre_norm=pre_norm,
            gate_residual=gate_residual,
            scale_residual=scale_residual,
            shift_tokens=shift_tokens,
            sandwich_norm=sandwich_norm,
            use_qk_norm_attn=use_qk_norm_attn,
            qk_norm_attn_seq_len=max_seq_len if use_qk_norm_attn else None,
            zero_init_branch_output=zero_init_branch_output,
            attn_talking_heads=attn_talking_heads,
            attn_on_attn=attn_on_attn,
            attn_gate_values=attn_gate_values,
            relative_spatial_pos_attr=relative_spatial_pos_attr,
            double_cross_attend=self.double_cross_attend
        )

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed
        self.token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed=l2norm_embed)
        self.pos_emb = (
            AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed=l2norm_embed)
            if (use_pos_emb and not attn_layers.has_pos_emb)
            else always(0)
        )
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()

        self.to_logits = (
            nn.Linear(dim, num_tokens)
            if not tie_embedding
            else lambda t: t @ self.token_emb.emb.weight.t()
        )

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.ordering = ordering

        self.spatial_position_emb = nn.ModuleList()
        if spatial_position_emb:
            assert spatial_position_emb in [
                e.value for e in TransformerSpatialConditioningType
            ], (
                f"spatial_position_emb must be one of the following {[e.value for e in TransformerSpatialConditioningType]},"
                f" but got {spatial_position_emb}."
            )
            axis = (0, 1, 2) if len(spatial_shape) == 3 else (0, 1)
            coord_channels = np.array(
                np.meshgrid(
                    *tuple(np.arange(0, s) for s in spatial_shape), indexing="ij"
                )
            )
            coord_channels = coord_channels[[s for s in axis]]

            for i in axis:
                spatial_indices_sequence = torch.from_numpy(
                    coord_channels[i, ...].flatten()
                )
                spatial_indices_sequence = self.ordering(spatial_indices_sequence)

                if (
                    spatial_position_emb
                    == TransformerSpatialConditioningType.FIXED.value
                ):
                    self.spatial_position_emb.append(
                        FixedSpatialPositionalEmbedding(
                            dim=dim, spatial_indices_sequence=spatial_indices_sequence
                        )
                    )
                elif (
                    spatial_position_emb
                    == TransformerSpatialConditioningType.ABSOLUTE.value
                ):
                    self.spatial_position_emb.append(
                        AbsoluteSpatialPositionalEmbedding(
                            dim=dim, spatial_indices_sequence=spatial_indices_sequence
                        )
                    )

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.token_emb.emb.weight, std=1e-5)
            for conditioning_emb in self.conditioning_emb:
                if conditioning_emb is not None:
                    nn.init.normal_(conditioning_emb.weight, std=1e-5)

            nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
            for spatial_position_emb in self.self.spatial_position_emb:
                nn.init.normal_(spatial_position_emb.emb.weight, std=1e-5)

            return

        nn.init.kaiming_normal_(self.token_emb.emb.weight)
        for conditioning_emb in self.conditioning_emb:
            if conditioning_emb is not None:
                nn.init.kaiming_normal_(conditioning_emb.weight)

    def forward(
        self,
        x: torch.Tensor,
        encoded_conditionings: torch.tensor = None,
        conditionings: Sequence[torch.Tensor] = None,
        return_embeddings: bool = False,
        mask: torch.Tensor = None,
        return_mems: bool = False,
        return_attn: bool = False,
        mems: torch.Tensor = None,
        **kwargs,
    ):
        b, n, device, num_mem = *x.shape, x.device, self.num_memory_tokens

        assert (
            n <= self.max_seq_len
        ), f"sequence length {n} must be less than the max sequence length {self.max_seq_len}"

        x = self.token_emb(x)

        for spatial_pos_emb in self.spatial_position_emb:
            x += spatial_pos_emb(x)

        if (
            conditionings
            and self.conditioning_type != TransformerConditioningType.NONE.value
        ):
            if (
                self.conditioning_type
                == TransformerConditioningType.BOSREPLACEMENT.value
            ):
                c = torch.unsqueeze(torch.zeros_like(x[:, 0, :]), 1)

                for idx, conditioning_emb in enumerate(self.conditioning_emb):
                    if conditioning_emb is not None:
                        c += conditioning_emb(conditionings[idx])
                    else:
                        c += torch.tile(
                            conditionings[idx][..., None], (1, 1, self.attn_layers.dim)
                        ).float()

                x[:, 0, :] = c[:, 0, :]
            elif self.conditioning_type == TransformerConditioningType.PREPENDING.value:
                for idx, conditioning_emb in enumerate(self.conditioning_emb):
                    if conditioning_emb is not None:
                        x = torch.cat((conditioning_emb(conditionings[idx]), x), dim=1)
                    else:
                        x = torch.cat(
                            (
                                torch.tile(
                                    conditionings[idx][..., None],
                                    (1, 1, self.attn_layers.dim),
                                ).float(),
                                x,
                            ),
                            dim=1,
                        )
            elif (
                self.conditioning_type == TransformerConditioningType.CROSSATTEND.value
            ):
                c = None

                for idx, conditioning_emb in enumerate(self.conditioning_emb):
                    if conditioning_emb is not None:
                        c = (
                            conditioning_emb(conditionings[idx])
                            if c is None
                            else torch.cat(
                                (conditioning_emb(conditionings[idx]), c), dim=1
                            )
                        )
                    else:
                        c = (
                            torch.tile(
                                conditionings[idx][..., None],
                                (1, 1, self.attn_layers.dim),
                            ).float()
                            if c is None
                            else torch.cat(
                                (
                                    torch.tile(
                                        conditionings[idx][..., None],
                                        (1, 1, self.attn_layers.dim),
                                    ).float(),
                                    c,
                                ),
                                dim=1,
                            )
                        )

                if encoded_conditionings is not None:
                    embedded_c = self.conditioning_encoded_emb(encoded_conditionings)

                    kwargs["context"] = [embedded_c, c]
                else:
                    kwargs["context"] = c

        if self.conditioning_type != TransformerConditioningType.CROSSATTEND.value:
            if encoded_conditionings is not None:
                embedded_c = self.conditioning_encoded_emb(encoded_conditionings)
                kwargs["context"] = embedded_c

        x = x + self.pos_emb(x)

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, "n d -> b n d", b=b)
            x = torch.cat((mem, x), dim=1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[: self.shift_mem_down], mems[self.shift_mem_down :]
            mems = [*mems_r, *mems_l]

        x, intermediates = self.attn_layers(
            x, mask=mask, mems=mems, return_hiddens=True, **kwargs
        )
        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        if (
            conditionings
            and self.conditioning_type != TransformerConditioningType.NONE.value
        ):
            if self.conditioning_type == TransformerConditioningType.PREPENDING.value:
                x = x[:, len(conditionings) :, :]

        out = self.to_logits(x) if not return_embeddings else x

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = (
                list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens)))
                if exists(mems)
                else hiddens
            )
            new_mems = list(
                map(lambda t: t[..., -self.max_mem_len :, :].detach(), new_mems)
            )
            return out, new_mems

        if return_attn:
            attn_maps = list(
                map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates)
            )
            return out, attn_maps

        return out
