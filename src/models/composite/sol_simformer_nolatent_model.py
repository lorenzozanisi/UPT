from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create
import torch
import einops 
from torch_geometric.utils import unbatch

class SolSimformerNolatentModel(CompositeModelBase):
    def __init__(
            self,
            model,
            conditioner=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        common_kwargs = dict(
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )

        # conditioner
        if conditioner is not None:
            self.conditioner = create(
                conditioner,
                model_from_kwargs,
                input_shape=self.input_shape,
                **common_kwargs,
            )        
        else:
            self.conditioner = None

        common_kwargs["static_ctx"]["condition_dim"] = self.conditioner.condition_embed.mlp[-2].out_features
        # encoder
        self.model = create(
            model,
            model_from_kwargs,
            input_shape=self.input_shape,
            input_features_shape=self.input_features_shape, 
            output_shape=self.output_shape,
            **common_kwargs,
        )

    
    @property
    def submodels(self):
        if self.conditioner is not None:
            return dict(
                conditioner=self.conditioner,
                model=self.model
            )
        else:
            return dict(
                model=self.model,
            )

    # noinspection PyMethodOverriding
    def forward(self, conditioning, mesh_pos, query_pos, input_features, batch_idx, unbatch_idx, unbatch_select):
        outputs = {}
        # encode data
        if self.conditioner is not None:
            condition = self.conditioner(conditioning) 
        else:
            condition = None

        x_hat = self.model(input_features=input_features, mesh_pos=mesh_pos, batch_idx=batch_idx, condition=condition)
        outputs["x_hat"] = x_hat


        x = self.model.norm(x_hat)
        x = self.model.output_proj(x)
        # dense tensor (batch_size, max_num_points, dim) -> sparse tensor (batch_size * num_points, dim)
        x = einops.rearrange(x, "batch_size max_num_points dim -> (batch_size max_num_points) dim")
        unbatched = unbatch(x, batch=unbatch_idx)
        x = torch.concat([unbatched[i] for i in unbatch_select])
        #exit(0)        
        outputs["x_hat"] = x

        return outputs
