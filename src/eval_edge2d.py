import wandb
from models import model_from_kwargs
from datasets.sol import Sol
from pathlib import Path, PurePath
from configs.static_config import StaticConfig
from datasets import dataset_from_kwargs
from providers.path_provider import PathProvider
from providers.dataset_config_provider import DatasetConfigProvider
from initializers.base.checkpoint_initializer import CheckpointInitializer
import yaml

import matplotlib
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
from scipy.interpolate import griddata




def plot(nvertp, rvertp, zvertp, korpg, fig, field, ax):
    """
    Plot the field on the mesh defined by nvertp, rvertp, zvertp, korpg.
    These can be found in the 'plotting_utils' group of each h5 file. 
    nvertp: number of vertices per polygon
    rvertp: radial coordinates of vertices
    zvertp: vertical coordinates of vertices
    korpg: polygon indices
    field: field values to plot, found in the 'targets_2d' group of each h5 file.
    fig: matplotlib figure object
    ax: matplotlib axes object
    """
    nump = len(nvertp)
    mesh=[]
    total_points = []
    for i in range(len(nvertp)):
        j = nvertp[i]
        points = np.transpose(np.concatenate(([rvertp[5*i:5*i+j]], [-zvertp[5*i:5*i+j]]), axis=0))
        total_points.append(points)
        polygon = Polygon(xy=points,closed= True)
        mesh.append(polygon)

    #fieldPolyg=field[korpg[korpg!=0]]
    fieldPolyg=np.zeros(nump)
    for i in range(nump):
        if korpg[i] > 0:
            fieldPolyg[korpg[i]-1]=field[i]

    assert len(fieldPolyg)== len(mesh)
    assert not np.isnan(fieldPolyg).any()
    p = PatchCollection(mesh,cmap='rainbow') #,norm=matplotlib.colors.SymLogNorm(linthresh=5e3))
    p.set_array(np.array(fieldPolyg))

    im = ax.add_collection(p)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Te (eV)')

    ax.set_xlabel("R(m)")
    ax.set_ylabel("Z(m)")
    ax.set_title("Te")
    total_points = np.vstack(total_points)
    ax.set_ylim(np.min(total_points[:,1]),np.max(total_points[:,1]))
    ax.set_xlim(np.min(total_points[:,0]),np.max(total_points[:,0]))

    return ax, cbar

print('Loading config')
#wandb_path = PurePath('/rds/project/iris_vol2/rds-ukaea-ap001/ir-zani1/UPT/UPT/checkpoints/stage1/95gurh8r/') # no conditioning

#wandb_path = PurePath('/rds/project/iris_vol2/rds-ukaea-ap001/ir-zani1/UPT/UPT/checkpoints/stage1/z0ltsm08/') # with conditioning
#model_name = PurePath('7lkn7nv8')
#model_name = PurePath('zmpaoss6') # with conditioning, all vars and xfield transport, new data
model_name = PurePath('g4wefv0p') # nolatent transformer 70M params
hp_resolved = PurePath('hp_resolved.yaml')
wandb_path = PurePath('/rds/project/iris_vol2/rds-ukaea-ap001/ir-zani1/UPT/UPT/checkpoints/stage1/')
model_path = wandb_path / model_name

cfg_path = model_path / hp_resolved
plots_path = Path('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/UPT/UPT/plots/') / model_name
plots_path.mkdir(parents=True, exist_ok=True)


with open(cfg_path, 'r') as f:
    cfg = yaml.safe_load(f)

# collator = cfg["datasets"]["train"]["collators"]
# model = cfg["model"]
# model_composite = model["kind"]
# encoder = model["encoder"]
# latent = model["latent"]
# encoder = model["encoder"]
print('preparing dataset')
static_config = StaticConfig(uri="static_config.yaml")
stage_name = cfg.get("stage_name", "default_stage")
stage_id = 0
path_provider = PathProvider(
    output_path=static_config.output_path,
    model_path=static_config.model_path,
    stage_name=stage_name,
    stage_id=stage_id,
    temp_path=static_config.temp_path,
)
dataset_config_provider = DatasetConfigProvider(
    global_dataset_paths=static_config.get_global_dataset_paths(),
    local_dataset_path=static_config.get_local_dataset_path(),
    data_source_modes=static_config.get_data_source_modes(),
)

print('trained on',cfg["datasets"]["train"]["conditioning_vars"])
#exit(0)
try:
    conditioning_vars = cfg["datasets"]["train"]["conditioning_vars"]
except:
    conditioning_vars = None

print('conditioning on:',conditioning_vars)
Sol = dataset_from_kwargs(
                dataset_config_provider=dataset_config_provider,
                path_provider=path_provider,
                **cfg["datasets"]["test"],
            )


model = model_from_kwargs(
    **cfg["model"],
    input_shape=(None,2), #trainer.input_shape,
    output_shape=(None,1),
    input_features_shape=(None,3),
    path_provider=path_provider,
 ) 


print('loading from checkpoint')
submodel_dict =  model.submodels
checkpoints_dir = model_path / "checkpoints"
model_kind = cfg["model"]["kind"]

for submodel_name, submodel in submodel_dict.items():
    prefix = f"{model_kind}.{submodel_name} "
    best_chkpt = "cp=best_model.loss.test.total model.th"
    chkpt_path = checkpoints_dir / f"{prefix}{best_chkpt}"
    #assert Path(chkpt_path).as_posix(), f'Path {chkpt_path} does not exist'[=]
    print(f'Loading model {chkpt_path}')
    submodel.load_state_dict(torch.load(chkpt_path)["state_dict"])
    print(submodel)    

df_conditions = Sol.conditions
print('evaluating')
test_idxs = np.random.choice(np.arange(0, len(df_conditions)), 50)
for idx in test_idxs:    
    electron_temp = Sol.getitem_target(idx=idx)
    korpg, nvertp, zvertp, rvertp, nump = Sol.getitem_grid_utils(idx=idx)
    input_mesh = Sol.getitem_mesh_pos(idx=idx)
    query_mesh = Sol.getitem_query_pos(idx=idx)
    input_features = Sol.getitem_input_features(idx=idx)


    if conditioning_vars is not None:
        conditions_float = []
        conditions_integers = []
        for key in conditioning_vars:
            item = getattr(Sol,f'getitem_{key}')(idx=idx)
            if item.dtype == torch.int8:
                conditions_integers.append(item)
            else:
                conditions_float.append(item)
        conditions_float = torch.stack(conditions_float).unsqueeze(-1)
        conditions_integers = torch.stack(conditions_integers).unsqueeze(-1)
        conditions = {'float': conditions_float, 'int': conditions_integers}
        # NOTE using iloc instad of loc as the index is not the same as that of the dataframe
        sim_path = getattr(Sol,'getitem_path')(idx=idx) 
        
    else:
        conditions = None

    temp_mean = Sol.scaling_stats["electron_temp_2d"]["mean"]
    temp_std = Sol.scaling_stats["electron_temp_2d"]["std"]

    out = model.forward(conditioning=conditions,
                        input_features=input_features,
                        mesh_pos=input_mesh,
                        query_pos=torch.unsqueeze(query_mesh, dim=1),
                        batch_idx=torch.zeros(input_mesh.size(0), dtype=torch.long), 
                        unbatch_idx=torch.zeros(input_mesh.size(0), dtype=torch.long), 
                        unbatch_select=[0]
                        )
    predicted_temp = out["x_hat"].squeeze().detach().numpy()
    electron_temp = electron_temp.detach().numpy()
    electron_temp = electron_temp*temp_std+temp_mean
    predicted_temp = predicted_temp*temp_std+temp_mean
    abserr = np.abs(predicted_temp - electron_temp)
    relerr = abserr / electron_temp
    print(f'abs error: {np.mean(abserr)}, rel error: {np.mean(relerr*100)}')  
    
    fig, ax = plt.subplots(1,3, figsize=(12,5))


    ax[0], _ = plot(nvertp, rvertp, zvertp, korpg, fig=fig, field=predicted_temp,  ax=ax[0]  )
    ax[1], _ = plot(nvertp, rvertp, zvertp, korpg, fig=fig, field=electron_temp,  ax=ax[1]   )
    ax[2], cbar = plot(nvertp, rvertp, zvertp, korpg, fig=fig, field=abserr,  ax=ax[2]   )

    cbar.set_label('Abs error')

    ax[0].set_title('Te predicted')
    ax[1].set_title('Te true')
    ax[2].set_title('Abs error')
    fig.suptitle(sim_path)
    fig.tight_layout()
    print('saving')
    fig.savefig(plots_path /f'{idx}.png')

