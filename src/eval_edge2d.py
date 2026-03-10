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
import matplotlib.colors as mcolors
import math
import sys
from scipy.interpolate import griddata


def first_n_significant_digits_float(x, n=4):
    """
    Returns the first n significant digits of x as a float, preserving magnitude.
    
    Args:
        x (float or int): The number to process.
        n (int): Number of significant digits to keep.
    
    Returns:
        float: Number rounded to n significant digits.
    """
    if x == 0:
        return 0.0
    
    magnitude = math.floor(math.log10(abs(x)))
    # Scale and round to n digits
    scaled = round(x / (10**magnitude) * (10**(n-1)))
    # Scale back to original magnitude
    return scaled * (10**(magnitude - (n-1)))

def plot_profiles(nvertp, rvertp, zvertp, korpg, fig, field, ax, label, norm=None):
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
        j = 4 #nvertp[i]
        points = np.transpose(np.concatenate(([rvertp[5*i:5*i+j]], [-zvertp[5*i:5*i+j]]), axis=0))
        total_points.append(points)
        polygon = Polygon(xy=points,closed= True)
        mesh.append(polygon)

    fieldPolyg=np.zeros(nump)
    for i in range(len(field)):
        if i>= len(korpg): continue
        if korpg[i] > 0:
            fieldPolyg[korpg[i]-1]=field[i]

    assert len(fieldPolyg)== len(mesh)
    assert not np.isnan(fieldPolyg).any()
    p = PatchCollection(mesh,cmap='viridis', norm=norm) #,norm=matplotlib.colors.SymLogNorm(linthresh=5e3))
    p.set_array(np.array(fieldPolyg))

    im = ax.add_collection(p)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)

    ax.set_xlabel("R(m)")
    ax.set_ylabel("Z(m)")

    total_points = np.vstack(total_points)
    ax.set_ylim(np.min(total_points[:,1]),np.max(total_points[:,1]))
    ax.set_xlim(np.min(total_points[:,0]),np.max(total_points[:,0]))

    return fig, ax,cbar

def r2_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)        # Residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
    r2 = 1 - ss_res / ss_tot
    return r2


def main(model_name='5px0ahwz', variable='carbon_radiation'):

    dict_name_mapping = {'deuterium_puff_values':r'$\Gamma_D \ [s^-1]$',
                        'power': 'Input power [MW]',
                        'puff_location_encoded':'Puff Location',
                        'electron_temp_2d':'Log10 Electron Temperature [eV]',
                        'carbon_radiation':r'Log10 Carbon Radiation $[Wm^{-3}]$',
                        'hydrogenic_radiation':r'Log10 Deuterium Radiation $[Wm^{-3}]$',
                        }
    dict_puff_mapping = {0:'LFS-D',1:'HFS-M',2:'LFS-V', 3:'PFR' }

    print('Loading config')

    hp_resolved = PurePath('hp_resolved.yaml')
    wandb_path = PurePath('/rds/project/iris_vol2/rds-ukaea-ap001/ir-zani1/UPT/UPT/checkpoints/stage1/')
    model_path = wandb_path / model_name
    cfg_path = model_path / hp_resolved
    plots_path = Path('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/UPT/UPT/plots/') / model_name
    plots_path.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    try:
        collator = cfg["datasets"]["train"]["collators"]
        model = cfg["model"]
        model_composite = model["kind"]
        encoder = model["encoder"]
        latent = model["latent"]
        encoder = model["encoder"]
    except:
        pass
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
    Sol.target_name = variable
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

                  
    print('evaluating')
    df_conditions = Sol.conditions
    #Sol.scaling_strategy = 'standardscaler'

    test_idxs = np.random.choice(np.arange(0, len(df_conditions)), 20)
    avg_abserr = []
    avg_relerr = []
    avg_r2 = []

    for idx in test_idxs:    
        true = Sol.getitem_target(idx=idx)
        korpg, nvertp, zvertp, rvertp, nump = Sol.getitem_grid_utils(idx=idx)
        input_mesh = Sol.getitem_mesh_pos(idx=idx)
        query_mesh = Sol.getitem_query_pos(idx=idx)
        input_features = Sol.getitem_input_features(idx=idx)

        conditions_dict = {}
        if conditioning_vars is not None:
            conditions_float = []
            conditions_integers = []
            for key in conditioning_vars:
                item = getattr(Sol,f'getitem_{key}')(idx=idx)
                if item.dtype == torch.int8:
                    conditions_integers.append(item)
                    item = item.item()
                else:
                    conditions_float.append(item)
                    item = first_n_significant_digits_float(item.item())

                conditions_dict[key] = item
            conditions_float = torch.stack(conditions_float).unsqueeze(-1)
            conditions_integers = torch.stack(conditions_integers).unsqueeze(-1)
            conditions = {'float': conditions_float, 'int': conditions_integers}
            # NOTE using iloc instad of loc as the index is not the same as that of the dataframe
            sim_path = getattr(Sol,'getitem_path')(idx=idx) 
            
        else:
            conditions = None

        scaling_stats_conditions = Sol.scaling_stats_conditions
        out = model.forward(conditioning=conditions,
                            input_features=input_features,
                            mesh_pos=input_mesh,
                            query_pos=torch.unsqueeze(query_mesh, dim=1),
                            batch_idx=torch.zeros(input_mesh.size(0), dtype=torch.long), 
                            unbatch_idx=torch.zeros(input_mesh.size(0), dtype=torch.long), 
                            unbatch_select=[0]
                            )
        predicted = out["x_hat"].squeeze().detach().numpy()
        true = true.detach().numpy()
        if Sol.scaling_strategy=='standardscaler':
            mean = Sol.scaling_stats[variable]["mean"]
            std = Sol.scaling_stats[variable]["std"]            
            true = true*std +mean #-1
            predicted = predicted*std+mean #)-1
        elif Sol.scaling_strategy=='minmaxscaler':
            maxval = Sol.scaling_stats[variable]["max"]
            minval = Sol.scaling_stats[variable]["min"]            
            true = true*(maxval-minval) + minval
            predicted = predicted*(maxval-minval) + minval
        predicted = np.where(predicted<0, 0, predicted) # set negative predictions to 0
        abserr = np.abs(predicted - true)
        relerr = np.abs(abserr / true)*100
        r2 = r2_score(true, predicted)
        avg_abserr.append(np.mean(abserr))
        avg_relerr.append(np.mean(relerr))
        avg_r2.append(np.mean(r2))
        print(f'abs error: {np.mean(abserr)}, % error: {np.mean(relerr)}')  

        relerr = np.where(relerr>20, 20, relerr) # cap percentage error at 20% for better visualization
        
        fig, ax = plt.subplots(1,4, figsize=(16,5))

        # vmin = min(np.min(true), np.min(predicted))
        # vmax = max(np.max(true), np.max(predicted))
        # norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        norm=None

        fig, ax[0], cbar0 = plot_profiles(nvertp, rvertp, zvertp, korpg, fig=fig, field=predicted,  ax=ax[0] , label=dict_name_mapping[variable] , norm=norm)
        fig, ax[1], cbar1 = plot_profiles(nvertp, rvertp, zvertp, korpg, fig=fig, field=true,  ax=ax[1]  , label=dict_name_mapping[variable], norm=norm)
        fig, ax[2], _ = plot_profiles(nvertp, rvertp, zvertp, korpg, fig=fig, field=relerr,  ax=ax[2]  , label='Percentage err', norm=None )
        fig, ax[3], _ = plot_profiles(nvertp, rvertp, zvertp, korpg, fig=fig, field=abserr,  ax=ax[3]  , label='Abs err', norm=None )



        ax[0].set_title('Predicted')
        ax[1].set_title('True')
        ax[2].set_title('Rel error')
        ax[3].set_title('Abs error')


        #puff_loc = f"Puff Location: {conditions_dict.get('puff_location')} "
        parts = []

        for key, value in conditions_dict.items():
            pretty_name = dict_name_mapping[key]

            if key == 'puff_location_encoded':
                value = dict_puff_mapping[value]
                
            if isinstance(value, float):
                value = first_n_significant_digits_float(scaling_stats_conditions[key]['mean'] + value*scaling_stats_conditions[key]['std'])
            parts.append(f"{pretty_name}: {value}")

        final_string = ", ".join(parts)    

        fig.suptitle(final_string)
        fig.tight_layout()
        print('saving')
        outpath = plots_path / str(variable)
        if not outpath.exists():
            outpath.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath/ f'{idx}.png')
        fig.clf()

        bins = np.linspace(0,np.max([true.max(), predicted.max()]), 50)
        plt.hist(true, bins=bins, alpha=0.5, label='True')
        plt.hist(predicted, bins=bins, alpha=0.5, label='Pred')
        plt.legend()
        plt.savefig(outpath/ f'{idx}_hist.png')
        plt.clf()

    results = {
        'avg_abserr': np.mean(avg_abserr),
        'avg_relerr': np.mean(avg_relerr),
        'avg_r2': np.mean(avg_r2)
    }
    print(results)
    with open(outpath / 'results.yaml', 'w') as f:
        yaml.dump(results, f)



if __name__ == "__main__":
    #main(variable='electron_temp_2d', model_name='bgug2cta')
    #main(variable='carbon_radiation', model_name='zaksw0dj')
    #main(variable='hydrogenic_radiation', model_name='2t44ajip')
    main(variable='electron_temp_2d', model_name='a024suto')


