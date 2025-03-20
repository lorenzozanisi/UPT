from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create


class Edge2dSimformerNognnModel(CompositeModelBase):
    def __init__(
            self,
            encoder,
            latent,
            decoder,
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
        if "condition_dim" in self.static_ctx.keys():
            self.conditioner = create(
                conditioner,
                model_from_kwargs,
                **common_kwargs,
                input_shape=self.input_shape,
            )        
        else:
            self.conditioner = None
        # encoder
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            **common_kwargs,
        )
        # latent
        self.latent = create(
            latent,
            model_from_kwargs,
            input_shape=self.encoder.output_shape,
            **common_kwargs,
        )
        # decoder
        self.decoder = create(
            decoder,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.latent.output_shape,
            output_shape=self.output_shape,
        )

    @property
    def submodels(self):
        return dict(
            conditioner=self.conditioner,
            encoder=self.encoder,
            latent=self.latent,
            decoder=self.decoder,
        )

    # noinspection PyMethodOverriding
    def forward(self, conditioning, mesh_pos, query_pos, batch_idx, unbatch_idx, unbatch_select):
        outputs = {}

        # encode data
        print('Conditioner...')
        if self.conditioner is not None:
            condition = self.conditioner(conditioning) 
        else:
            condition = None

        print('Encoder...')
        encoded = self.encoder(mesh_pos=mesh_pos, batch_idx=batch_idx, condition=condition)

        # propagate
        #Condition TODO
        print('Latent...')
        propagated = self.latent(encoded, condition=condition) 

        # decode
        #Condition TODO
        print('Decoder...')
        x_hat = self.decoder(propagated, condition=condition, query_pos=query_pos, unbatch_idx=unbatch_idx, unbatch_select=unbatch_select)
        outputs["x_hat"] = x_hat

        return outputs
