from functools import partial

import einops
import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import PerceiverBlock, DitPerceiverBlock
from kappamodules.vit import DitBlock, VitBlock
from torch import nn
from torch_geometric.utils import unbatch

from models.base.single_model_base import SingleModelBase


class LagrangianTransformerPerceiver(SingleModelBase):
    def __init__(
            self,
            dim,
            depth,
            num_attn_heads,
            init_weights="xavier_uniform",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.init_weights = init_weights

        # input/output shape
        num_channels, _  = self.output_shape
        seqlen, input_dim = self.input_shape
        ndim = self.data_container.get_dataset().metadata["dim"]

        # input projection
        self.input_proj = LinearProjection(input_dim, dim)

        # blocks
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = VitBlock
        self.blocks = nn.ModuleList([
            block_ctor(dim=dim, num_heads=num_attn_heads, init_weights=init_weights)
            for _ in range(self.depth)
        ])

        # query tokens (create them from a positional embedding)
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.query_mlp = nn.Sequential(
            LinearProjection(dim, dim * 4),
            nn.GELU(),
            LinearProjection(dim * 4, dim * 4),
            nn.GELU(),
            LinearProjection(dim * 4, dim),
        )

        # latent to pixels
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitPerceiverBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PerceiverBlock
        self.perceiver = block_ctor(dim=dim, num_heads=num_attn_heads, init_weights=init_weights)
        self.pred = LinearProjection(dim, num_channels)

    def forward(self, x, query_pos, unbatch_idx, unbatch_select, static_tokens=None, condition=None):
        assert x.ndim == 3

        # input projection
        x = self.input_proj(x)

        # apply blocks
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        for blk in self.blocks:
            x = blk(x, **block_kwargs)

        # create query
        pos_embed = self.pos_embed(query_pos)
        query = self.query_mlp(pos_embed)

        # latent to pixels
        x = self.perceiver(q=query, kv=x, **block_kwargs)
        x = self.pred(x)

        # dense tensor (batch_size, max_num_points, dim) -> sparse tensor (batch_size * num_points, dim)
        x = einops.rearrange(x, "batch_size max_num_points dim -> (batch_size max_num_points) dim")
        unbatched = unbatch(x, batch=unbatch_idx)
        x = torch.concat([unbatched[i] for i in unbatch_select])

        return x
