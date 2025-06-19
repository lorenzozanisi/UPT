import einops
import scipy
import os
import shutil
from functools import lru_cache
import logging

import meshio
import h5py
import numpy as np
import torch
from kappautils.param_checking import to_3tuple, to_2tuple
from torch_geometric.nn.pool import radius, radius_graph
import pandas as pd
from distributed.config import barrier, is_data_rank0
import pickle
from .base.dataset_base import DatasetBase


class Sol(DatasetBase):
    def __init__(
            self,
            split,
            radius_graph_r=None,
            radius_graph_max_num_neighbors=None,
            num_input_points_ratio=None,
            num_query_points_ratio=None,
            grid_resolution=None,
            num_supernodes=None,
            standardize_query_pos=False,
            concat_pos_to_sdf=False,
            global_root=None,
            local_root=None,
            seed=None,
            conditioning_vars=None,
            conditioning_vars_fname=None,
            smoke_test=False,
            **kwargs,
    ):
        """
        Edge2D dataset for the Sol-based models.
        Args:
            split (str): Split of the dataset, either "train" or "test".
            radius_graph_r (float): Radius for the graph construction.
            radius_graph_max_num_neighbors (int): Maximum number of neighbors for the graph construction.
            num_input_points_ratio (Union[None, Tuple[float, float]]): Ratio of input points to sample from the mesh.
            num_query_points_ratio (float): Ratio of query points to sample from the mesh.
            grid_resolution (Union[None, Tuple[int, int]]): Resolution of the grid for the interpolated field.
            num_supernodes (Union[None, int]): Number of supernodes to sample from the mesh.
            standardize_query_pos (bool): Standardize query positions to [-1, 1].
            concat_pos_to_sdf (bool): Concatenate position to the sdf features.
            global_root (Path): Global root of the dataset.
            local_root (Path): Local root of the dataset.
            seed (int): Seed for the random number generator.
            conditioning_vars (List[str]): List of conditioning variables. Passed in yaml file
            conditioning_vars_fname (str): Name of the pickle file containing the conditioning variables. Passed in the yaml file.
            **kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.conditioning_vars_fname = conditioning_vars_fname
        self.conditioning_vars = conditioning_vars
        self.split = split
        self.radius_graph_r = radius_graph_r
        self.radius_graph_max_num_neighbors = radius_graph_max_num_neighbors or int(1e10)
        self.num_supernodes = num_supernodes
        self.seed = seed
        if num_input_points_ratio is None:
            self.num_input_points_ratio = None
        else:
            self.num_input_points_ratio = to_2tuple(num_input_points_ratio) # TODO check what this does
        self.num_query_points_ratio = num_query_points_ratio
        if grid_resolution is not None:
            self.grid_resolution = to_3tuple(grid_resolution) #TODO check what this does
        else:
            self.grid_resolution = None

        self.scale = 200
        self.standardize_query_pos = standardize_query_pos
        self.concat_pos_to_sdf = concat_pos_to_sdf

        global_root, local_root = self._get_roots(global_root, local_root, "sol")
        if local_root is None:
            # load data from global_root
            self.source_root = global_root / "preprocessed"
            self.logger.info(f"data_source (global): '{self.source_root}'")
        else:
            # load data from local_root
            self.source_root = local_root / "sol"
            if is_data_rank0():
                # copy data from global to local
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{self.source_root}'")
                if not self.source_root.exists():
                    self.logger.info(
                        f"copying {(global_root / 'preprocessed').as_posix()} "
                        f"to {(self.source_root / 'preprocessed').as_posix()}"
                    )
                    shutil.copytree(global_root / "preprocessed", self.source_root / "preprocessed")
            self.source_root = self.source_root / "preprocessed"

            barrier()
        assert self.source_root.exists(), f"'{self.source_root.as_posix()}' doesn't exist"
        assert self.source_root.name == "preprocessed", f"'{self.source_root.as_posix()}' is not preprocessed folder"

        with open(self.source_root / "stats.pkl", "rb") as f:
            self.scaling_stats = pickle.load(f)

        # discover uris
     #   sim_idxs = []
    #    self.uris = []
        # for name in os.listdir(self.source_root):
        #     if name!='.' and not (name.endswith('pkl') or name.endswith('csv') or name.endswith('json')):
        #   #      sim_idxs.append(name.split("_")[1].split(".")[0])  # extract simulation index from the name                 
        #         uri = self.source_root / name
        #         self.uris.append(uri)
        # logging.info(f'Discovered {len(self.uris)} uris')
        #sorted_idxs = np.argsort(np.array(sim_idxs, dtype=int))
        #self.uris = [self.uris[idx] for idx in sorted_idxs]

        if self.conditioning_vars_fname is not None:
            self.conditions = self.load_conditions()
        else:
            logging.info(f"Could not find conditioning_vars_fname, defaulting to no conditioning")
            self.conditions = None
        
        # if split == "train":
        #     train_idxs = self.conditions.query('train==True').index
        #     self.uris = [self.uris[train_idx] for train_idx in train_idxs]
        # elif split == "test":
        #     test_idxs = self.conditions.query('test==False').index
        #     self.uris = [self.uris[test_idx] for test_idx in test_idxs] # self.TEST_INDICES]

        # else:
        #     raise NotImplementedError
                        
        if split == "train":
            self.conditions = self.conditions.query('train==True')
        elif split == "test":
            self.conditions = self.conditions.query('test==True')
        else:
            raise NotImplementedError
        
        self.uris = []
        # filter uris for indices that satisfy the conditions in the conditions dataframe
        # uris are now indexed not by the index of the conditions dataframe but by their position in the list
        if self.conditions is not None:
            tmp_uris = []
            for idx in self.conditions.index:
                uri = self.source_root / f'simulation_{idx}.h5'
                self.uris.append(uri)
            self.uris = tmp_uris
        else:
            # --- use all
            for name in os.listdir(self.source_root):
                if name!='.' and not (name.endswith('pkl') or name.endswith('csv') or name.endswith('json')):
            #      sim_idxs.append(name.split("_")[1].split(".")[0])  # extract simulation index from the name                 
                    uri = self.source_root / name
                    self.uris.append(uri)
        logging.info(f'Discovered {len(self.uris)} uris')
        
        self.conditions = self.scale_conditions()

    def __len__(self):
        return len(self.uris)
    
    # noinspection PyUnusedLocal
    # TODO: to implement class for loading, scaling and unscaling conditions
    def load_conditions(self):
        conditions = pd.read_pickle(self.source_root / self.conditioning_vars_fname)
        return conditions
                           
    def scale_conditions(self):
        metadata = set(self.conditions.columns) - set(self.conditioning_vars)
        df_meta = self.conditions[list(metadata)]
        conditions = self.conditions[self.conditioning_vars]
        index = conditions.index
        mean = conditions.values.mean(axis=0)
        scaled = conditions.values-mean
        std = scaled.std(axis=0)
        conditions = pd.DataFrame(scaled / std, columns=self.conditioning_vars, dtype=np.float16, index=index)
        conditions = pd.merge(df_meta, conditions, left_index=True, right_index=True)
        return conditions
        
    # def getitem_target(self, idx, ctx=None):
    #     with h5py.File(self.uris[idx], 'r') as h5file:
    #         tmp = h5file[f"targets2d"]["target"][:]
    #     tmp = torch.from_numpy(tmp)
    #     tmp -= self.mean["target"]
    #     tmp /= self.std["target"]
    #     return tmp 

    # --- TODO: to be updated to actual target 
    def getitem_target(self, idx, ctx=None):
        with h5py.File(self.uris[idx], 'r') as h5file:
            tmp = np.array(list(h5file[f"targets2d"]["electron_temp_2d"]))
        tmp = torch.from_numpy(tmp)
        tmp -= self.scaling_stats["electron_temp_2d"]["mean"]
        tmp /= self.scaling_stats["electron_temp_2d"]["std"]
        return tmp     

    # --- TODO: to be updated to actual target
    def getshape_target(self):
        # with h5py.File(self.uris[0], 'r') as h5file:
        #     tmp = h5file[f"targets2d"]["electron_temp_2d"][:]
        return None, 1
    
    def getitem_psin(self, idx, ctx=None):
        with h5py.File(self.uris[idx], 'r') as h5file:
            tmp = np.array(list(h5file[f"inputs2d"]["psin"]))
        tmp = torch.from_numpy(tmp)
        tmp -= self.scaling_stats["psin"]["mean"]
        tmp /= self.scaling_stats["psin"]["std"]
        return tmp     

    def getitem_b_toroidal(self, idx, ctx=None):
        with h5py.File(self.uris[idx], 'r') as h5file:
            tmp = np.array(list(h5file[f"inputs2d"]["b_toroidal"]))
        tmp = torch.from_numpy(tmp)
        tmp -= self.scaling_stats["b_toroidal"]["mean"]
        tmp /= self.scaling_stats["b_toroidal"]["std"]
        return tmp     
    
    def getitem_sh(self, idx, ctx=None):
        with h5py.File(self.uris[idx], 'r') as h5file:
            tmp = np.array(list(h5file[f"inputs2d"]["sh"]))
        tmp = torch.from_numpy(tmp)
        tmp -= self.scaling_stats["sh"]["mean"]
        tmp /= self.scaling_stats["sh"]["std"]
        return tmp     
        
    def getnames_conditioning_vars(self):
        return self.conditioning_vars
    
    # NOTE: using iloc instead of loc as the index is not the same as the index of the conditions dataframe
    def getitem_path(self, idx, ctx=None):
        return self.conditions.iloc[idx]["path"]
    
    def getitem_connection_length(self, idx, ctx=None): 
        tmp = self.conditions.iloc[idx,"connection_length"]
        tmp = torch.tensor(tmp)
        return tmp

    def getitem_deuterium_puff_values(self, idx, ctx=None):
        tmp = self.conditions.iloc[idx]["deuterium_puff_values"]
        tmp = torch.tensor(tmp)
        return tmp        

    def getitem_psep(self, idx, ctx=None): 
        tmp = self.conditions.iloc[idx]["psep"]
        tmp = torch.tensor(tmp)
        return tmp

    def getitem_pumped_neutral_flux(self, idx, ctx=None): 
        tmp = self.conditions.iloc[idx]["pumped_neutral_flux"]
        tmp = torch.tensor(tmp)
        return tmp

    def getitem_inner_avg_albedo(self, idx, ctx=None): 
        tmp = self.conditions.iloc[idx]["inner_avg_albedo"]
        tmp = torch.tensor(tmp)
        return tmp            

    def getitem_outer_avg_albedo(self, idx, ctx=None): 
        tmp = self.conditions.iloc[idx]["outer_avg_albedo"]
        tmp = torch.tensor(tmp)
        return tmp            
    
    def getitem_particle_flux_omp(self, idx, ctx=None): 
        tmp = self.conditions.iloc[idx]["particle_flux_omp"]
        tmp = torch.tensor(tmp)
        return tmp            

    def getitem_pumped_neutral_flux(self, idx, ctx=None): 
        tmp = self.conditions.iloc[idx]["pumped_neutral_flux"]
        tmp = torch.tensor(tmp)
        return tmp            

    def getitem_flux_expansion(self, idx, ctx=None): 
        tmp = self.conditions.iloc[idx]["flux_expansion"]
        tmp = torch.tensor(tmp)
        return tmp                        

    def getitem_strike_point_poloidal_angle(self, idx, ctx=None): 
        tmp = self.conditions.iloc[idx]["strike_point_poloidal_angle"]
        tmp = torch.tensor(tmp)
        return tmp                        
    
    def getitem_grid_utils(self, idx):
        with h5py.File(self.uris[idx], "r") as h5file:
            korpg = h5file["plotting_utils"]["korpg"][:]
            nvertp = h5file["plotting_utils"]["nvertp"][:]
            zvertp = h5file["plotting_utils"]["zvertp"][:]
            rvertp = h5file["plotting_utils"]["rvertp"][:]
            nump = h5file["plotting_utils"]["np"][:]            
        return korpg, nvertp, zvertp, rvertp, nump    

    # noinspection PyUnusedLocal
    # --- Only used when using grid-based stuff such as GINO
    def getitem_grid_pos(self, idx=None, ctx=None):
        if ctx is not None and "grid_pos" in ctx:
            return ctx["grid_pos"]
        # generate positions for a regular grid (e.g. for GINO encoder)
        assert self.grid_resolution is not None
        x_linspace = torch.linspace(0, self.scale, self.grid_resolution[0])
        y_linspace = torch.linspace(0, self.scale, self.grid_resolution[1])
        # generate positions (grid_resolution[0] * grid_resolution[1], 2)
        meshgrid = torch.meshgrid(x_linspace, y_linspace)
        grid_pos = torch.stack(meshgrid).flatten(start_dim=1).T
        #
        if ctx is not None:
            assert "grid_pos" not in ctx
            ctx["grid_pos"] = grid_pos
        return grid_pos

    # --- Only used when using grid-based stuff such as GINO
    def getitem_mesh_to_grid_Sols(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.radius_graph_r is not None
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        grid_pos = self.getitem_grid_pos(idx, ctx=ctx)
        # create graph between mesh and regular grid points
        Sols = radius(
            x=mesh_pos,
            y=grid_pos,
            r=self.radius_graph_r,
            max_num_neighbors=self.radius_graph_max_num_neighbors,
        ).T
        # Sols is (num_points, 2)
        return Sols
    
    # --- Only used when using grid-based stuff such as GINO
    def getitem_grid_to_query_Sols(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.radius_graph_r is not None
        query_pos = self.getitem_query_pos(idx, ctx=ctx)
        grid_pos = self.getitem_grid_pos(idx, ctx=ctx)
        # create graph between mesh and regular grid points
        Sols = radius(
            x=grid_pos,
            y=query_pos,
            r=self.radius_graph_r,
            max_num_neighbors=int(1e10),
        ).T
        # Sols is (num_points, 2)
        return Sols

    def getitem_mesh_pos(self, idx, ctx=None):
        if ctx is not None and "mesh_pos" in ctx:
            return ctx["mesh_pos"]
        mesh_pos = self.getitem_all_pos(idx, ctx=ctx)
        # sample mesh points
        if self.num_input_points_ratio is not None:
            if self.split == "test":
                assert self.seed is not None
            if self.seed is not None:
                # deterministically downsample for evaluation
                generator = torch.Generator().manual_seed(self.seed + int(idx))
            else:
                generator = None
            # get number of samples
            if self.num_input_points_ratio[0] == self.num_input_points_ratio[1]:
                # fixed num_input_points_ratio
                end = int(len(mesh_pos) * self.num_input_points_ratio[0])
            else:
                # variable num_input_points_ratio
                lb, ub = self.num_input_points_ratio
                num_input_points_ratio = torch.rand(size=(1,), generator=generator).item() * (ub - lb) + lb
                end = int(len(mesh_pos) * num_input_points_ratio)
            # uniform sampling
            perm = torch.randperm(len(mesh_pos), generator=generator)[:end]
            mesh_pos = mesh_pos[perm]
        if ctx is not None:
            ctx["mesh_pos"] = mesh_pos
        return mesh_pos

    def getitem_all_pos(self, idx, ctx=None):
        if ctx is not None and "all_pos" in ctx:
            return ctx["all_pos"]
        with h5py.File(self.uris[idx], 'r') as h5file:
            r = h5file["mesh"]["rmesh2d"][:]
            z = h5file["mesh"]["zmesh2d"][:]
        r = torch.from_numpy(r)#-self.scaling_stats["rmesh2d"]["min"] #this ensures that the minimum value is 0
        z = torch.from_numpy(z)#-self.scaling_stats["zmesh2d"]["min"] #this ensures that the minimum value is 0

        #logging.info(f'Beofre: Rmin, rmax, zmin, zmax {r.min()}, {r.max()}, {z.min()}, {z.max()}')
        r = (r-self.scaling_stats["rmesh2d"]["min"]) / (self.scaling_stats["rmesh2d"]["max"] - self.scaling_stats["rmesh2d"]["min"]) * self.scale
        z = (z-self.scaling_stats["zmesh2d"]["min"]) / (self.scaling_stats["zmesh2d"]["max"] - self.scaling_stats["zmesh2d"]["min"]) * self.scale
        # r.sub_(self.scaling_stats["rmesh2d"]["min"]).div_(self.scaling_stats["rmesh2d"]["max"] - self.scaling_stats["rmesh2d"]["min"]).mul_(self.scale)
        # z.sub_(self.scaling_stats["zmesh2d"]["min"]).div_(self.scaling_stats["zmesh2d"]["max"] - self.scaling_stats["zmesh2d"]["min"]).mul_(self.scale)
        #logging.info(f'After: Rmin, rmax, zmin, zmax {r.min()}, {r.max()}, {z.min()}, {z.max()}')

        all_pos = torch.stack([r,z], dim=1)
        #logging.info(f'all_pos has shape {all_pos.shape}')


     #   all_pos = torch.from_numpy(np.vstack((r,z)).T)
        ##all_pos = torch.load(self.uris[idx] / "mesh_points.th")
        # rescale for sincos positional embedding
     #   all_pos.sub_(self.scaling).div_(self.domain_max - self.domain_min).mul_(self.scale)
       
        assert torch.all(0 <=all_pos)
        assert torch.all(all_pos <= self.scale)
        if ctx is not None:
            ctx["all_pos"] = all_pos
        return all_pos

    def getitem_query_pos(self, idx, ctx=None):
        if ctx is not None and "query_pos" in ctx:
            return ctx["query_pos"]
        query_pos = self.getitem_all_pos(idx, ctx=ctx)
        # sample query points
        if self.num_query_points_ratio is not None:
            if self.split == "test":
                assert self.seed is not None
            if self.seed is not None:
                # deterministically downsample for evaluation
                generator = torch.Generator().manual_seed(self.seed + int(idx))
            else:
                generator = None
            # get number of samples
            end = int(len(query_pos) * self.num_query_points_ratio)
            # uniform sampling
            perm = torch.randperm(len(query_pos), generator=generator)[:end]
            query_pos = query_pos[perm]
        # shift query_pos to [-1, 1] (required for torch.nn.functional.grid_sample)
        if self.standardize_query_pos:
            query_pos = query_pos / (self.scale / 2) - 1
        if ctx is not None:
            ctx["query_pos"] = query_pos
        return query_pos

    def _get_generator(self, idx):
        if self.split == "test":
            return torch.Generator().manual_seed(int(idx) + (self.seed or 0))
        if self.seed is not None:
            return torch.Generator().manual_seed(int(idx) + self.seed)
        return None

    # noinspection PyUnusedLocal
    def getitem_mesh_Sols(self, idx, ctx=None):
        assert self.radius_graph_r is not None
        # load mesh positions
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        if self.num_supernodes is None:
            # create graph
            Sols = radius_graph(
                x=mesh_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
                loop=True,
            )
        else:
            # select supernodes
            generator = self._get_generator(idx)
            perm = torch.randperm(len(mesh_pos), generator=generator)[:self.num_supernodes]
            supernodes_pos = mesh_pos[perm]
            # create Sols: this can include self-loop or not depending on how many neighbors are found.
            # if too many neighbors are found, neighbors are selected randomly which can discard the self-loop
            Sols = radius(
                x=mesh_pos,
                y=supernodes_pos,
                r=self.radius_graph_r,
                max_num_neighbors=self.radius_graph_max_num_neighbors,
            )
            # correct supernode index
            Sols[0] = perm[Sols[0]]
        return Sols.T

    # noinspection PyUnusedLocal
    # --- TODO
    def getitem_sdf(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert all(self.grid_resolution[0] == grid_resolution for grid_resolution in self.grid_resolution[1:])
        sdf = torch.load(self.uris[idx] / f"sdf_res{self.grid_resolution[0]}.th")
        # check that sdf features were generated with correct positions by checking the distance to the nearest point
        # from the domain minimum/maximum
        # mesh_pos = torch.load(self.uris[idx] / "mesh_points.th")
        # minpoint_dists = (self.domain_min[None, :] - mesh_pos).norm(p=2, dim=1)
        # maxpoint_dists = (self.domain_max[None, :] - mesh_pos).norm(p=2, dim=1)
        # assert torch.allclose(sdf[0, 0, 0], minpoint_dists.min()), f"{sdf[0, 0, 0]} != {minpoint_dists.min()}"
        # assert torch.allclose(sdf[-1, -1, -1], maxpoint_dists.min()), f"{sdf[-1, -1, -1]} != {maxpoint_dists.min()}"
        if self.concat_pos_to_sdf:
            # add position to sdf (GINO uses this for interpolated FNO model)
            x_linspace = torch.linspace(-1, 1, self.grid_resolution[0])
            y_linspace = torch.linspace(-1, 1, self.grid_resolution[1])
            grid_pos = torch.meshgrid(x_linspace, y_linspace)
            # stack features (models expect dim_last format)
            sdf = torch.stack([sdf, *grid_pos], dim=-1)
        else:
            sdf = sdf.unsqueeze(-1)
        return sdf


    def getitem_interpolated(self, idx, ctx=None):
        assert self.grid_resolution is not None
        assert self.standardize_query_pos
        mesh_pos = self.getitem_mesh_pos(idx, ctx=ctx)
        # generate grid positions (these are different than getitem_gridpos because interpolate requires xy indexing)
        # it should be the same if indexing=ij since the mapping and inverse mapping consider the change in indexing
        # but for consistency with scipy.interpolate xy was chosen
        x_linspace = torch.linspace(0, self.scale, self.grid_resolution[0])
        y_linspace = torch.linspace(0, self.scale, self.grid_resolution[1])
        grid_pos = torch.meshgrid(x_linspace, y_linspace)

        grid = torch.from_numpy(
            scipy.interpolate.griddata(
                mesh_pos.unbind(1),
                torch.ones_like(mesh_pos),
                grid_pos,
                method="linear",
                fill_value=0.,
            ),
        ).float()

        # check for correctness of interpolation
        # import matplotlib.pyplot as plt
        # import os
        # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        # plt.scatter(mesh_pos[:, 0], mesh_pos[:, 1])
        # plt.show()
        # plt.clf()
        # plt.imshow(grid.sum(dim=2).sum(dim=2), origin="lower")
        # plt.show()
        # plt.clf()
        # import torch.nn.functional as F
        # grid = einops.rearrange(grid, "h w d dim -> 1 dim h w d")
        # query_pos = self.getitem_query_pos(idx, ctx=ctx)
        # query_pos = einops.rearrange(query_pos, "num_points ndim -> 1 num_points 1 1 ndim")
        # mesh_values = F.grid_sample(input=grid, grid=query_pos, align_corners=False).squeeze(-1)
        # plt.scatter(*query_pos.squeeze().unbind(1), c=mesh_values[0, 0, :, 0])
        # plt.show()
        # plt.clf()

        return grid