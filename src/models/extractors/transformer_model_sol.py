from functools import partial

import torch
from kappamodules.layers import LinearProjection, ContinuousSincosEmbed
from kappamodules.transformer import DitBlock, PrenormBlock,PerceiverPoolingBlock, Mlp, PerceiverPoolingBlock, DitPerceiverPoolingBlock
from torch import nn
from torch_geometric.utils import unbatch
import einops
from torch_geometric.utils import to_dense_batch


from models.base.single_model_base import SingleModelBase


class TransformerModelSol(SingleModelBase):
    def __init__(
            self,
            dim,
            depth,
            num_attn_heads,
            drop_path_rate=0.0,
            drop_path_decay=True,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            cond_dim=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero

        # input/output shape
        assert len(self.input_shape) == 2
        assert len(self.output_shape) == 2
        _, ndim = self.input_shape # dimension of the mesh (ie 2 or 3)
        _, n_features = self.input_features_shape # number of features per node        
        seqlen, output_dim = self.output_shape

        self.input_proj = LinearProjection(n_features, dim, init_weights=init_weights, bias=False)
        self.output_proj = LinearProjection(dim, output_dim, init_weights=init_weights)
        self.norm = nn.LayerNorm(dim, eps=1e-6) # if use_last_norm else nn.Identity()
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)

        # input and output shapes are set in the base class by the create function
        # blocks
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PrenormBlock
        if drop_path_decay:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth
        self.blocks = nn.ModuleList([
            block_ctor(
                dim=dim,
                num_heads=num_attn_heads,
                drop_path=dpr[i],
                init_weights=init_weights,
                init_last_proj_zero=init_last_proj_zero,
            )
            for i in range(self.depth)
        ])


    #def forward(self, x, condition=None, static_tokens=None):
    def forward(self, input_features, mesh_pos, batch_idx, condition=None):
       # assert x.ndim == 3
        x = self.pos_embed(mesh_pos)
        input_features = self.input_proj(input_features)
        x = x + input_features

        # concat static tokens
        # if static_tokens is not None:
        #     x = torch.cat([static_tokens, x], dim=1)

        x, mask = to_dense_batch(x, batch_idx)
        print(f"SolPerceiver: x shape: {x.shape}, mask shape: {mask.shape if mask is not None else None}")
        if torch.all(mask):
            mask = None
        else:
            # add dimensions for num_heads and query (keys are masked)
            mask = einops.rearrange(mask, "batchsize num_nodes -> batchsize 1 1 num_nodes")

        # apply blocks
        blk_kwargs = {}
        if condition is not None:
            blk_kwargs["cond"] = condition
            
        print('x shape is', x.shape)
        for blk in self.blocks:
            x = blk(x, **blk_kwargs)
            print('x shape is', x.shape)

       # exit(0)
        

        # remove static tokens
        # if static_tokens is not None:
        #     num_static_tokens = static_tokens.size(1)
        #     x = x[:, num_static_tokens:]
        x = self.norm(x)
        x = self.output_proj(x)
        print('x shape before einops', x.shape)
        # dense tensor (batch_size, max_num_points, dim) -> sparse tensor (batch_size * num_points, dim)
        x = einops.rearrange(x, "batch_size max_num_points dim -> (batch_size max_num_points) dim")
        print('x shape after einops', x.shape)
        # unbatched = unbatch(x, batch=batch_idx)
        # x = torch.concat([unbatched[i] for i in unbatch_select])
        #print(f"predict shape", x.shape, "output shape", self.output_shape)
        #exit(0)
        return x
