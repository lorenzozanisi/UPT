
from typing import Optional, Sequence, Union

from einops import rearrange
import torch
import logging
from torch import nn
from kappamodules.functional.pos_embed import get_sincos_1d_from_seqlen

class ContinuousConditionEmbed(nn.Module):
    '''From https://git.bioinf.jku.at/setinek/plasmamodelling/-/blob/main/models/utils.py?ref_type=heads'''
    def __init__(
        self,
        dim: int,
        n_cond: int,
        max_wavelength: int = 10_000,
        init_weights: Optional[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_cond = n_cond
        #self.ndim_padding = dim % n_cond
        #dim_per_ndim = (dim - self.ndim_padding) // n_cond
        #self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        #self.padding = self.ndim_padding + self.sincos_padding * n_cond
        cond_per_wave = self.dim  // n_cond
        assert cond_per_wave > 0
        self.register_buffer(
            "omega",
            1.0 / max_wavelength ** (torch.arange(0, cond_per_wave, 2) / cond_per_wave),
        )
        self.cond_dim =  dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, self.cond_dim),
            nn.SiLU(),
        )


        if init_weights is not None:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):

        if init_weights == "torch" or init_weights is None:
            pass
        elif init_weights == "xavier_uniform":
            self.mlp.apply(seq_weight_init(nn.init.xavier_uniform_))
        elif init_weights in ["truncnormal", "truncnormal002"]:
            self.mlp.apply(seq_weight_init(nn.init.trunc_normal_))
        else:
            raise NotImplementedError

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim == 1:
            cond = cond.unsqueeze(-1)
        
        cond = cond.view((cond.shape[-1], -1))
        logging.info(f'cond shape {cond.shape}')
        #   assert self.n_cond == cond.shape[-1], f"{self.n_cond} != {cond.shape[-1]}"
        #print(cond, self.omega)
        #exit(0)
        out = cond.unsqueeze(-1).type(self.omega.dtype) @ self.omega.unsqueeze(0)
        emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
        logging.info(f'cond 1 shape {cond.shape}')
        emb = rearrange(emb, "... ncond cdim -> ... (ncond cdim)")
        logging.info(f'emb 2 shape {cond.shape}')
        # if self.padding > 0:
        #     padding = torch.zeros(
        #         *emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype
        #     )
        #     emb = torch.concat([emb, padding], dim=-1)
        emb = self.mlp(emb)
        return emb

def seq_weight_init(weight_init_fn, bias_init_fn=None):
    if bias_init_fn is None:
        bias_init_fn = nn.init.zeros_

    def _apply(m):
        if isinstance(m, nn.Linear):
            weight_init_fn(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                bias_init_fn(m.bias)

    return _apply



    # conditioning_int = self.embed_integer(conditioning['int'].long()) # to be expanded to more than one integer
    #     conditioning_float = self.condition_embed(conditioning['float'])
    #     print('conditionign shape ', conditioning_int.shape, conditioning_float.shape)
    #     embedded = torch.cat((conditioning_float, conditioning_int.squeeze(dim=0)), dim=0)