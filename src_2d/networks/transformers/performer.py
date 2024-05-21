from src.utils.constants import (
    TransformerConditioningType,
    TransformerSpatialConditioningType,
)
from src.utils.transformer import (
    AbsoluteSpatialPositionalEmbedding,
    FixedSpatialPositionalEmbedding,
)

from typing import Optional, Tuple, Union, Sequence
import numpy as np

to_exclude = ['Performer']
from performer_pytorch.performer_pytorch import *
for name in to_exclude:
    del globals()[name]


from torch import nn
from src.networks.transformers.img2seq_ordering import Ordering
from src.networks.transformers.transformer import TransformerBase


class PreShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim = -1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask = mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim = -1)
        return self.fn(x, **kwargs)


class DoubleAttention(nn.Module):
    def __init__(
            self,
            dim,
            causal=False,
            heads=8,
            dim_head=64,
            local_heads=0,
            local_window_size=256,
            nb_features=None,
            feature_redraw_interval=1000,
            generalized_attention=False,
            kernel_fn=nn.ReLU(),
            dropout=0.,
            no_projection=False,
            qkv_bias=False,
            attn_out_bias=True,
            number_in_parallel=1,
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, nb_features, causal=causal,
                                            generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                            no_projection=no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size=local_window_size, causal=causal, autopad=True, dropout=dropout,
                                         look_forward=int(not causal),
                                         rel_pos_emb_config=(dim_head, local_heads)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=qkv_bias)

        self.to_q_2 = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_k_2 = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.to_v_2 = nn.Linear(dim, inner_dim, bias=qkv_bias)

        self.to_out = nn.Linear(inner_dim, dim, bias=attn_out_bias)
        self.to_out_2 = nn.Linear(inner_dim, dim, bias=attn_out_bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb=None, context=None, mask=None, context_mask=None, **kwargs):

        cross_attend = exists(context)

        num_layers = 1 if context is None else len(context)

        for l in range(num_layers):
            b, n, _, h, gh = *x.shape, self.heads, self.global_heads

            context_val = x if context is None else context[l]
            # context_val = x if context is None else torch.unsqueeze(context[l],dim=0)
            context_mask = default(context_mask, mask) if not cross_attend else context_mask[
                l] if context_mask is not None else None


            if l == 0:
                q, k, v = self.to_q(x), self.to_k(context_val), self.to_v(context_val)
            else:
                q, k, v = self.to_q_2(x), self.to_k_2(context_val), self.to_v_2(context_val)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
            (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

            attn_outs = []

            if not empty(q):
                if exists(context_mask):
                    global_mask = context_mask[l][:, None, :, None]
                    v.masked_fill_(~global_mask, 0.)

                if exists(pos_emb) and not cross_attend:
                    q, k = apply_rotary_pos_emb(q, k, pos_emb)

                out = self.fast_attention(q, k, v)
                attn_outs.append(out)

            if not empty(lq):
                assert not cross_attend, 'local attention is not compatible with cross attention'
                out = self.local_attn(lq, lk, lv, input_mask=mask)
                attn_outs.append(out)

            out = torch.cat(attn_outs, dim=1)
            out = rearrange(out, 'b h n d -> b n (h d)')

            if l == 0:
                x = self.to_out(out)
                x = self.dropout(x)
            else:
                x = self.to_out_2(out)
                x = self.dropout(x)

            if num_layers == 1:
                return x

        return x

class DoubleCrossAttention(DoubleAttention):
    def forward(self, *args, context=None, **kwargs):
        assert exists(context), 'cross attention should receive context'
        return super().forward(*args, context=context, **kwargs)


class BasePerformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        ff_glu = False,
        ff_dropout = 0.,
        attn_dropout = 0.,
        cross_attend = False,
        no_projection = False,
        auto_check_redraw = True,
        qkv_bias = True,
        attn_out_bias = True,
        shift_tokens = False,
        double_cross_attend = False
    ):
        super().__init__()
        layers = nn.ModuleList([])
        local_attn_heads = cast_tuple(local_attn_heads)
        local_attn_heads = local_attn_heads * depth if len(local_attn_heads) == 1 else local_attn_heads
        assert len(local_attn_heads) == depth, 'tuple specifying number of local attention heads per depth must be equal to the total depth'
        assert all(map(lambda n: n >= 0 and n <= heads, local_attn_heads)), 'local attention head value must be less than the total number of heads'

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _, local_heads in zip(range(depth), local_attn_heads):

            attn = SelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, local_heads = local_heads, local_window_size = local_window_size, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)
            ff = Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1)

            if shift_tokens:
                shift = (0, 1) if causal else (-1, 0, 1)
                attn, ff = map(lambda t: PreShiftTokens(shift, t), (attn, ff))

            attn, ff = map(wrapper_fn, (attn, ff))
            layers.append(nn.ModuleList([attn, ff]))

            if not cross_attend:
                continue

            if double_cross_attend:
                layers.append(nn.ModuleList([
                    wrapper_fn(DoubleCrossAttention(dim, heads = heads, dim_head = dim_head, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)),
                    wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
                ]))
            else:
                layers.append(nn.ModuleList([
                    wrapper_fn(CrossAttention(dim, heads = heads, dim_head = dim_head, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_out_bias = attn_out_bias)),
                    wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
                ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence

        route_attn = ((True, False),) * depth * (2 if cross_attend else 1)
        route_context = ((False, False), (True, False)) * depth
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}
        context_route_map = {'context': route_context, 'context_mask': route_context} if cross_attend else {}
        self.net = execute_type(layers, args_route = {**attn_route_map, **context_route_map})

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.proj_updater = ProjectionUpdater(self.net, feature_redraw_interval)

    def fix_projection_matrices_(self):
        self.proj_updater.feature_redraw_interval = None

    def forward(self, x, **kwargs):
        if self.auto_check_redraw:
            self.proj_updater.redraw_projections()
        return self.net(x, **kwargs)


class Performer(TransformerBase):
    """
        NOTE: All tensor logic assumes the following ordering [Batch, Length, Channel]
    """

    def __init__(
        self,
        *,
        num_tokens: int,
        max_seq_len: int,
        dim: int,
        depth: int,
        heads: int,
        ordering: Ordering,
        dim_head: int = 64,
        local_attn_heads: int = 0,
        local_window_size: int = 256,
        causal: bool = True,
        ff_mult: int = 4,
        nb_features: Optional[int] = None,
        feature_redraw_interval: int = 1000,
        reversible: bool = False,
        ff_chunks: int = 1,
        ff_glu: bool = False,
        emb_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        generalized_attention: bool = False,
        kernel_fn: torch.nn.Module = nn.ReLU(),
        use_scalenorm: bool = False,
        use_rezero: bool = False,
        cross_attend: bool = False,
        no_projection: bool = False,
        tie_embed: bool = False,
        rotary_position_emb: bool = False,
        fixed_position_emb: bool = False,
        axial_position_emb: bool = False,
        axial_position_shape: Tuple[int, int] = None,
        auto_check_redraw: bool = True,
        qkv_bias: bool = False,
        attn_out_bias: bool = False,
        spatial_position_emb: str = None,
        spatial_shape: Union[Tuple[int, int], Tuple[int, int, int]] = None,
        conditioning_num_tokens: Optional[Tuple[int, ...]] = None,
        conditioning_type: str = TransformerConditioningType.NONE.value,
        num_tokens_enc: int = 0,
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        # Accounting for the number of prepended conditionings
        self.max_seq_len = max_seq_len + (
            len(conditioning_num_tokens)
            if conditioning_num_tokens
            and conditioning_type == TransformerConditioningType.PREPENDING.value
            else 0
        )
        self.token_emb = nn.Embedding(num_tokens, dim)

        assert (
            0 <= sum([rotary_position_emb, fixed_position_emb, axial_position_emb]) <= 1
        ), (
            f"rotary_position_emb, fixed_position_emb and axial_position_emb are exclusive, but received "
            f"{rotary_position_emb} {fixed_position_emb} and {axial_position_emb}."
        )
        if rotary_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, self.max_seq_len)
            self.layer_pos_emb = FixedPositionalEmbedding(dim_head, self.max_seq_len)
        elif fixed_position_emb:
            self.pos_emb = FixedPositionalEmbedding(dim, self.max_seq_len)
            self.layer_pos_emb = Always(None)
        elif axial_position_emb:
            axial_position_shape = default(
                axial_position_shape, (math.ceil(self.max_seq_len / 64), 64)
            )
            self.pos_emb = AxialPositionalEmbedding(dim, axial_position_shape)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, self.max_seq_len)
            self.layer_pos_emb = Always(None)

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

        self.dropout = nn.Dropout(emb_dropout)

        self.performer = BasePerformer(
            dim,
            depth,
            heads,
            dim_head,
            local_attn_heads,
            local_window_size,
            causal,
            ff_mult,
            nb_features,
            feature_redraw_interval,
            reversible,
            ff_chunks,
            generalized_attention,
            kernel_fn,
            use_scalenorm,
            use_rezero,
            ff_glu,
            ff_dropout,
            attn_dropout,
            self.cross_attend,
            no_projection,
            auto_check_redraw,
            qkv_bias,
            attn_out_bias,
            double_cross_attend=self.double_cross_attend
        )
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(
        self,
        x: torch.tensor,
        encoded_conditionings: torch.tensor = None,
        conditionings: Sequence[torch.Tensor] = None,
        token_masking: torch.tensor = None,
        return_encodings: bool = False,
        **kwargs,
    ):

        if token_masking is not None:
            kwargs["mask"] = token_masking
        if conditionings is not None:
            kwargs["context_mask"] = token_masking

        b, n, device = *x.shape, x.device
        assert (
            n <= self.max_seq_len
        ), f"sequence length {n} must be less than the max sequence length {self.max_seq_len}"

        x = self.token_emb(x)

        for spatial_pos_emb in self.spatial_position_emb:
            x += spatial_pos_emb(x)

        layer_pos_emb = self.layer_pos_emb(x)

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

        if (self.conditioning_type != TransformerConditioningType.CROSSATTEND.value) or (conditionings is None) :
            if encoded_conditionings is not None:
                embedded_c = self.conditioning_encoded_emb(encoded_conditionings)
                kwargs["context"] = embedded_c

        x += self.pos_emb(x)

        x = self.dropout(x)

        x = self.performer(x, pos_emb=layer_pos_emb, **kwargs)

        # norm and to logits
        x = self.norm(x)

        if (
            conditionings
            and self.conditioning_type != TransformerConditioningType.NONE.value
        ):
            if self.conditioning_type == TransformerConditioningType.PREPENDING.value:
                x = x[:, len(conditionings) :, :]

        if return_encodings:
            return x

        if exists(self.to_out):
            return self.to_out(x)

        return x @ self.token_emb.weight.t()
