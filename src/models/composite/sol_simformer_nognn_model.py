from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create


class SolSimformerNognnModel(CompositeModelBase):
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
        print('Encoder:', self.encoder)
        print('Latent:', self.latent)
        print('Decoder:', self.decoder)
        if self.conditioner is not None:
            print('Conditioner:', self.conditioner)

    @property
    def submodels(self):
        if self.conditioner is not None:
            return dict(
                conditioner=self.conditioner,
                encoder=self.encoder,
                latent=self.latent,
                decoder=self.decoder,
            )
        else:
            return dict(
                encoder=self.encoder,
                latent=self.latent,
                decoder=self.decoder,
            )

    # noinspection PyMethodOverriding
    def forward(self, conditioning, mesh_pos, query_pos, batch_idx, unbatch_idx, unbatch_select):
        outputs = {}

        # encode data
        if self.conditioner is not None:
            condition = self.conditioner(conditioning) 
        else:
            condition = None

        encoded = self.encoder(mesh_pos=mesh_pos, batch_idx=batch_idx, condition=condition)

        # propagate
        propagated = self.latent(encoded, condition=condition) 

        # decode
        x_hat = self.decoder(propagated, condition=condition, query_pos=query_pos, unbatch_idx=unbatch_idx, unbatch_select=unbatch_select)
        outputs["x_hat"] = x_hat

        return outputs
