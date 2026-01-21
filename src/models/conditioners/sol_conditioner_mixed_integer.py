from kappamodules.init import init_xavier_uniform_zero_bias, init_truncnormal_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
import torch
from utils.condition_embed import ContinuousConditionEmbed
from models.base.single_model_base import SingleModelBase


class SolConditioner(SingleModelBase):
    def __init__(self, dim, cond_dim=None, init_weights="xavier_uniform", **kwargs):
        super().__init__(**kwargs)
        #self.conditioning_vars = self.data_container.get_dataset().getnames_conditioning_vars()
        self.dim = dim
        self.n_cond = cond_dim # or dim * 4
        self.init_weights = init_weights
        self.static_ctx["condition_dim"] = self.n_cond 

        #cond_dim = len(self.conditioning_vars)
        self.condition_embed = ContinuousConditionEmbed(dim=dim, n_cond=cond_dim)
        self.embed_integers = torch.nn.Embedding(3, dim)  
        

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
        conditioning_int = self.embed_puff_location(conditioning['int']) # to be expanded to more than one integer
        conditioning_float = self.condition_embed(conditioning['float'])
        embedded = conditioning_float + conditioning_int
        
        return embedded
