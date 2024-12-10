import wandb
from models import model_from_kwargs
from datasets.edge2d import Edge2D
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

# def plot():
#     nump = len(nvertp)
#     fig, ax = plt.subplots(1,1)
#     mesh=[]
#     total_points = []
#     for i in range(len(nvertp)):
#     #  color='r' if nvertp.data[i]==4 else 'g'
#     j = nvertp[i]
#     points = np.transpose(np.concatenate(([rvertp[5*i:5*i+j]], [-zvertp[5*i:5*i+j]]), axis=0))
#     total_points.append(points)
#     polygon = Polygon(xy=points,closed= True)
#     mesh.append(polygon)


#     field=te
#     #fieldPolyg=field[korpg[korpg!=0]]
#     fieldPolyg=np.zeros(nump)
#     for i in range(nump):
#         if korpg[i] > 0:
#             fieldPolyg[korpg[i]-1]=field[i]

#     assert len(fieldPolyg)== len(mesh)
#     assert not np.isnan(fieldPolyg).any()
#     p = PatchCollection(mesh,cmap='rainbow') #,norm=matplotlib.colors.SymLogNorm(linthresh=5e3))
#     p.set_array(np.array(fieldPolyg))

#     ax.add_collection(p)


#     plt.xlabel("R(m)")
#     plt.ylabel("Z(m)")
#     plt.title("Te")
#     total_points = np.vstack(total_points)
#     plt.ylim(np.min(total_points[:,1]),np.max(total_points[:,1]))
#     plt.xlim(np.min(total_points[:,0]),np.max(total_points[:,0]))

print('Loading config')
wandb_path = PurePath('/rds/project/iris_vol2/rds-ukaea-ap001/ir-zani1/UPT/UPT/checkpoints/stage1/tfa0okoj/')
hp_resolved = PurePath('hp_resolved.yaml')
cfg_path = wandb_path / hp_resolved

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
edge2d = dataset_from_kwargs(
                dataset_config_provider=dataset_config_provider,
                path_provider=path_provider,
                **cfg["datasets"]["test"],
            )
#dataset = Edge2D(**cfg["datasets"]["test"]) # -- how does the model ingest data to make a prediction? MAybe just load a specific example and feed it straight to the model
print('preparing model')
model = model_from_kwargs(
    **cfg["model"],
    input_shape=(None,2), #trainer.input_shape,
    output_shape=(None,1),
    path_provider=path_provider,
 ) 


print('loading from checkpoint')
submodel_dict =  model.submodels
checkpoints_dir = wandb_path / "checkpoints"
model_kind = cfg["model"]["kind"]

for submodel_name, submodel in submodel_dict.items():
    prefix = f"{model_kind}.{submodel_name} "
    best_chkpt = "cp=best_model.loss.test.total model.th"
    chkpt_path = checkpoints_dir / f"{prefix}{best_chkpt}"
    #assert Path(chkpt_path).as_posix(), f'Path {chkpt_path} does not exist'
    submodel.load_state_dict(torch.load(chkpt_path)["state_dict"])
    
        #print(f"loading from {chkpt_path}")
# print(checkpoints_dir)
# checkpoints = os.listdir(checkpoints_dir)

# for key, submodel in submodel_dict.items():
#     print(f'loading model {key} from {checkpoints_dir}')
#     submodel_checkpoint_idx = np.where( np.array([this_file.find(key) for this_file in checkpoints])!=-1)[0][0]
#     submodel_checkpoint = wandb_path / checkpoints[submodel_checkpoint_idx]
#     submodel.load_state_dict(torch.load(submodel_checkpoint))
