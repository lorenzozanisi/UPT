from functools import cached_property

import torch
from kappadata.wrappers import ModeWrapper
from torch import nn
from torch_scatter import segment_csr

from callbacks.online_callbacks.update_output_callback import UpdateOutputCallback
from datasets.collators.rans_simformer_nognn_collator import RansSimformerNognnCollator
from losses import loss_fn_from_kwargs
from utils.factory import create
from .base.sgd_trainer import SgdTrainer


class RansSimformerNognnTrainer(SgdTrainer):
    def __init__(self, loss_function, max_batch_size=None, **kwargs):
        # automatic batchsize is not supported with mesh data
        disable_gradient_accumulation = max_batch_size is None
        super().__init__(
            max_batch_size=max_batch_size,
            disable_gradient_accumulation=disable_gradient_accumulation,
            **kwargs,
        )
        self.loss_function = create(loss_function, loss_fn_from_kwargs, update_counter=self.update_counter)

    @cached_property
    def input_shape(self):
        dataset, collator = self.data_container.get_dataset("train", mode="mesh_pos")
        assert isinstance(collator.collator, RansSimformerNognnCollator)
        mesh_pos, _ = dataset[0]
      
        # mesh_pos has shape (num_points, ndim)
        assert mesh_pos.ndim == 2 and 2 <= mesh_pos.size(1) <= 3
        return None, mesh_pos.size(1)

    @cached_property
    def output_shape(self):
        # pressure is predicted
        return None, 1

    @cached_property
    def dataset_mode(self):
        return "pressure mesh_pos query_pos"

    def get_trainer_model(self, model):
        return self.Model(model=model, trainer=self)

    class Model(nn.Module):
        def __init__(self, model, trainer):
            super().__init__()
            self.model = model
            self.trainer = trainer

        def to_device(self, item, batch):
            data = ModeWrapper.get_item(mode=self.trainer.dataset_mode, item=item, batch=batch)
            data = data.to(self.model.device, non_blocking=True)
            return data

        def prepare(self, batch):
            batch, ctx = batch
            return dict(
                mesh_pos=self.to_device(item="mesh_pos", batch=batch),
                query_pos=self.to_device(item="query_pos", batch=batch),
                batch_idx=ctx["batch_idx"].to(self.model.device, non_blocking=True),
                unbatch_idx=ctx["unbatch_idx"].to(self.model.device, non_blocking=True),
                unbatch_select=ctx["unbatch_select"].to(self.model.device, non_blocking=True),
                target=self.to_device(item="pressure", batch=batch),
            )

        def forward(self, batch, reduction="mean"):
            data = self.prepare(batch)
            target = data.pop("target")

            # forward pass
            model_outputs = self.model(**data)
            loss = self.trainer.loss_function(
                prediction=model_outputs["x_hat"],
                target=target,
                reduction=reduction,
            )

            # accumulate losses of points
            if reduction == "mean_per_sample":
                _, ctx = batch
                query_batch_idx = ctx["query_batch_idx"].to(self.model.device, non_blocking=True)
                # indptr is a tensor of indices betweeen which to aggregate
                # i.e. a tensor of [0, 2, 5] would result in [src[0] + src[1], src[2] + src[3] + src[4]]
                indices, counts = query_batch_idx.unique(return_counts=True)
                # first index has to be 0
                padded_counts = torch.zeros(len(indices) + 1, device=counts.device, dtype=counts.dtype)
                padded_counts[indices + 1] = counts
                indptr = padded_counts.cumsum(dim=0)
                loss = segment_csr(src=loss, indptr=indptr, reduce="mean")

            return dict(total=loss, x_hat=loss), {}
