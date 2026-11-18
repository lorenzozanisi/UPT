import einops
import torch
import torch_geometric
from torch import nn

from .continuous_sincos_embed import ContinuousSincosEmbed
from .linear_projection import LinearProjection

import torch_scatter


class SupernodePooling(nn.Module):
    """Supernode pooling layer with the following default settings:
    - Messages from incoming nodes are averaged (aggregation="mean")
    - After the message passing, the original positional embedding of the supernode is concated to the supernode
      followed by downprojection to the original hidden dimension
    The permutation of the supernodes is preserved through the message passing (contrary to the UPT code).
    Additionally, radius is used instead of radius_graph, which is more efficient.
    Args:
        radius (float): Radius around each supernode. From points within this radius, messages are passed to the
          supernode.
        hidden_dim (int): Hidden dimension for positional embeddings, messages and the resulting output vector.
        input_dim (int): Number of input features (use None if only positions are used).
        ndim (int): Number of positional dimension (e.g., ndim=2 for a 2D position, ndim=3 for a 3D position)
        max_degree (int): Maximum degree of the radius graph.
        init_weights (str): Weight initialization of linear layers.
        readd_supernode_pos (bool): If true, the absolute positional encoding of the supernode is concated to the
          supernode vector after message passing and linearly projected back to hidden_dim.
        aggregation (str): Aggregation for message passing ("mean" or "sum").
    """

    def __init__(
        self,
        hidden_dim: int,
        ndim: int,
        input_dim: int | None = None,
        radius: float | None = None,
        max_degree: int = 32,
        init_weights: str = "truncnormal002",
        readd_supernode_pos: bool = True,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.radius = radius
        self.max_degree = max_degree
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.init_weights = init_weights
        self.readd_supernode_pos = readd_supernode_pos
        self.aggregation = aggregation
        self.input_dim = input_dim

        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim)
        if input_dim is None:
            self.input_proj = None
        else:
            self.input_proj = LinearProjection(input_dim, hidden_dim, init_weights=init_weights)
        self.message = nn.Sequential(
            LinearProjection(hidden_dim * 2, hidden_dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(hidden_dim, hidden_dim, init_weights=init_weights),
        )
        if readd_supernode_pos:
            self.proj = LinearProjection(2 * hidden_dim, hidden_dim, init_weights=init_weights)
        else:
            self.proj = None
        self.output_dim = hidden_dim

    def forward(
        self,
        input_pos: torch.Tensor,
        supernode_idx: torch.Tensor,
        input_feat: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        assert input_pos.ndim == 2
        assert input_feat is None or input_feat.ndim == 2
        assert supernode_idx.ndim == 1

        # radius graph
        if batch_idx is None:
            batch_y = None
        else:
            batch_y = batch_idx[supernode_idx]
        edges = torch_geometric.nn.pool.radius(
            x=input_pos,
            y=input_pos[supernode_idx],
            r=self.radius,
            max_num_neighbors=self.max_degree,
            batch_x=batch_idx,
            batch_y=batch_y,
        )
        # remap dst indices
        dst_idx, src_idx = edges.unbind()
        dst_idx = supernode_idx[dst_idx]

        # create message
        x = self.pos_embed(input_pos)
        if self.input_dim is not None:
            assert input_feat is not None
            x = x + self.input_proj(input_feat)
        supernode_pos_embed = x[supernode_idx]
        x = torch.concat([x[src_idx], x[dst_idx]], dim=1)
        x = self.message(x)

        # accumulate messages
        # indptr is a tensor of indices betweeen which to aggregate
        # i.e. a tensor of [0, 2, 5] would result in [src[0] + src[1], src[2] + src[3] + src[4]]
        dst_indices, counts = dst_idx.unique_consecutive(return_counts=True)
        assert torch.all(supernode_idx == dst_indices)
        # first index has to be 0
        # NOTE: padding for target indices that don't occour is not needed as self-loop is always present
        padded_counts = torch.zeros(len(counts) + 1, device=counts.device, dtype=counts.dtype)
        padded_counts[1:] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = torch_scatter.segment_csr(src=x, indptr=indptr, reduce=self.aggregation)

        # sanity check: dst_indices has len of batch_size * num_supernodes and has to be divisible by batch_size
        # if num_supernodes is not set in dataset this assertion fails
        if batch_idx is None:
            batch_size = 1
        else:
            batch_size = batch_idx.max() + 1
            assert dst_indices.numel() % batch_size == 0

        # convert to dense tensor (dim last)
        x = einops.rearrange(
            x,
            "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
            batch_size=batch_size,
        )

        if self.readd_supernode_pos:
            supernode_pos_embed = einops.rearrange(
                supernode_pos_embed,
                "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
                batch_size=len(x),
            )
            # concatenate input and supernode embeddings
            x = torch.concat([x, supernode_pos_embed], dim=-1)
            x = self.proj(x)

        return x