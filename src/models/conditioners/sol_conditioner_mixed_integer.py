from kappamodules.init import init_xavier_uniform_zero_bias, init_truncnormal_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
import torch
from utils.condition_embed import ContinuousConditionEmbed
from models.base.single_model_base import SingleModelBase
import logging

class SolConditionerMixedInteger(SingleModelBase):
    def __init__(self, dim, cond_dim=None, init_weights="xavier_uniform", **kwargs):
        super().__init__(**kwargs)
        #self.conditioning_vars = self.data_container.get_dataset().getnames_conditioning_vars()
        self.dim = dim
        self.n_cond = cond_dim # or dim * 4
        self.init_weights = init_weights
        self.static_ctx["condition_dim"] = self.n_cond 
        logging.info(f'n_cond is {self.n_cond}')


        self.condition_embed = ContinuousConditionEmbed(dim=dim, n_cond=self.n_cond-1)
        out_embed_dim = self.condition_embed.mlp[-2].out_features
        self.embed_integer = torch.nn.Embedding(4, out_embed_dim)  
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(out_embed_dim*2, out_embed_dim),
            torch.nn.SiLU(),
        )        
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
        elif self.init_weights == "truncnormal":
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError

    def forward(self, conditioning):
        # checks + preprocess
        # assert timestep.numel() == len(timestep)
        # assert velocity.numel() == len(velocity)
        # timestep = timestep.flatten()
        # velocity = velocity.view(-1, 1).float()
        # # for rollout timestep is simply initialized as 0 -> repeat to batch dimension
        # if timestep.numel() == 1:
        #     timestep = timestep.repeat(velocity.numel())
        # embed
        # embedded = zeros(self.cond_dim).to(self.device)
        # assert len(self.conditioning_vars) == len(conditioning)
        # for cond_var in self.conditioning_vars:
        #     condition = getattr(self,cond_var)(conditioning[cond_var])
        #     embedded += getattr(self,cond_var+"_mlp")(condition)
        conditioning_int = self.embed_integer(conditioning['int'].long()) # to be expanded to more than one integer
        conditioning_float = self.condition_embed(conditioning['float'])
        print('conditionign shape ', conditioning_int.shape, conditioning_float.shape)
        embedded = torch.cat((conditioning_float, conditioning_int.squeeze(dim=0)), dim=-1)
        print('embedded concat', embedded.shape)
        embedded = self.mlp(embedded)

        #embedded = conditioning_float + conditioning_int
        logging.info(f'embedded dimension {embedded.shape}')
        return embedded#+conditioning_int
