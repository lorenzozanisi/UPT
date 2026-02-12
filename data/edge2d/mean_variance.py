import h5py
import glob
import numpy as np
import matplotlib.pylab as plt
from functools import partial
from multiprocessing import Pool
import json
import pickle as pkl
import pandas as pd

def process(file,h5_group,var_name):

    with h5py.File(file,'r') as f:
        tmp = np.array(list(f[h5_group][var_name]))
    if "density" in var_name:
        tmp /= 1e19

    return tmp

def main():
    """
    Calculate mean and std of each variable in the dataset
    and store it in a pickle file in a standardised location
    """
    target_names = ['electron_density_2d',
                    'electron_temp_2d',
                    'ion_density_2d',
                    'ion_temp_2d',
                    'neutral_atom_density_2d',
                    'neutral_atom_temperature_2d',
                    'neutral_molecule_density_2d',
                    'neutral_molecule_temperature_2d']
    input_names = ['b_toroidal',
                'psin',
                'sh',
                'd_perp',
                'chi_i',
                'chi_e']
    coords = ['rmesh2d',
            'zmesh2d']

    # with open('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/UPT/UPT/data/edge2d/files_ok.pkl','rb') as f:
    #     files = pkl.load(f)
    #files = glob.glob('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/UPT/UPT/data_store/sol/mastu/preprocessed/*.h5')        
    df = pd.read_pickle('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/UPT/UPT/data_store/sol/mastu/preprocessed/matched_full_dataframe.pkl')
    files = df['h5path'].values
    stats = {}

    for input_name in input_names:
        partial_process = partial(process, h5_group='inputs2d', var_name=input_name)
        with Pool(52) as pool:
            var = pool.map(partial_process, files)
        var = np.hstack(var)
        stats[input_name] = {'mean':var.mean(), 'std':var.std()}
    
    for var_name in target_names:
        print(var_name)
        partial_process = partial(process, h5_group='targets2d', var_name=var_name)
        with Pool(52) as pool:
            var = pool.map(partial_process, files)
        var = np.hstack(var)
        stats[var_name] = {'mean':var.mean(), 'std':var.std()}


    for coord_name in coords:
        print(coord_name)
        partial_process = partial(process, h5_group='mesh', var_name=coord_name)
        with Pool(52) as pool:
            var = pool.map(partial_process, files)
        var = np.hstack(var)
        stats[coord_name] = {'max':var.max(), 'min':var.min()}
       
        print(stats[coord_name]) 
    
        

    with open('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/UPT/UPT/data_store/sol/mastu/preprocessed/stats.pkl', 'wb') as f:
        pkl.dump(stats, f)


if __name__=='__main__':
    main()