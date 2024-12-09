# conda create --name open3d python=3.9
# pip install open3d
# pip install meshio
# pip install torch
# pip install tempfile
import os
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import meshio
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="e.g. /data/shapenet_car/training_data")
    parser.add_argument("--dst", type=str, required=True, help="e.g. /data/shapenet_car/preprocessed")
    return vars(parser.parse_args())


def sdf(mesh, resolution):
    quads = mesh.cells_dict["quad"]

    idx = np.flatnonzero(quads[:, -1] == 0)
    out0 = np.empty((quads.shape[0], 2, 3), dtype=quads.dtype)

    out0[:, 0, 1:] = quads[:, 1:-1]
    out0[:, 1, 1:] = quads[:, 2:]

    out0[..., 0] = quads[:, 0, None]

    out0.shape = (-1, 3)

    mask = np.ones(out0.shape[0], dtype=bool)
    mask[idx * 2 + 1] = 0
    quad_to_tri = out0[mask]

    cells = [("triangle", quad_to_tri)]
    print(cells)

    new_mesh = meshio.Mesh(mesh.points, cells)

    with tempfile.NamedTemporaryFile(delete=True, suffix=".ply") as tf:
        new_mesh.write(tf, file_format="ply")
        open3d_mesh = o3d.io.read_triangle_mesh(tf.name)
    open3d_mesh = o3d.t.geometry.TriangleMesh.from_legacy(open3d_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(open3d_mesh)

    domain_min = torch.tensor([-2.0, -1.0, -4.5])
    domain_max = torch.tensor([2.0, 4.5, 6.0])
    tx = np.linspace(domain_min[0], domain_max[0], resolution)
    ty = np.linspace(domain_min[1], domain_max[1], resolution)
    tz = np.linspace(domain_min[2], domain_max[2], resolution)
    grid = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(np.float32)
    return torch.from_numpy(scene.compute_signed_distance(grid).numpy()).float()


def main(src, dst):
    src = Path(src).expanduser()
    assert src.exists(), f"'{src.as_posix()}' doesnt exist"
    assert src.name == "training_data"
    dst = Path(dst).expanduser()
    # assert not dst.exists(), f"'{dst.as_posix()}' exist"
    print(f"src: {src.as_posix()}")
    print(f"dst: {dst.as_posix()}")

    # collect uris for samples
    uris = []
    for i in range(9):
        param_uri = src / f"param{i}"
        for name in sorted(os.listdir(param_uri)):
            # param folders contain .npy/.py/txt files
            if "." in name:
                continue
            potential_uri = param_uri / name
            assert potential_uri.is_dir()
            uris.append(potential_uri)
    print(f"found {len(uris)} samples")

    # .vtk files contains points that dont belong to the mesh -> filter them out
    mesh_point_counts = []
    for uri in tqdm(uris):
        print("Processing ", uri)
        reluri = uri.relative_to(src)
        out = dst / reluri
        out.mkdir(exist_ok=True, parents=True)

        # filter out mesh points that are not part of the shape
        mesh = meshio.read(uri / "quadpress_smpl.vtk")
        print(uri,mesh)
        assert len(mesh.cells) == 1
        cell_block = mesh.cells[0]
        assert cell_block.type == "quad"
        unique = np.unique(cell_block.data)
        print(unique)
        mesh_point_counts.append(len(unique))
        mesh_points = torch.from_numpy(mesh.points[unique]).float() # What does this do? ***
        pressure = torch.from_numpy(np.load(uri / "press.npy")[unique]).float()

        torch.save(mesh_points, out / "mesh_points.th")
        print(mesh_points)
        
        torch.save(pressure, out / "pressure.th")

        # -- generate sdf
        for resolution in [32, 40, 48, 64, 80]:
            torch.save(sdf(mesh, resolution=resolution), out / f"sdf_res{resolution}.th")

    print("fin")


if __name__ == "__main__":
    main(**parse_args())



# 
# Basically this means that len(mesh.points)>len(unique) that is there are more mesh points than IDs used to define the mesh elelements (like quadrilatera).
# See chatGPT answer below.

# ***The difference between `len(mesh.points)` and `len(unique)` lies in how the points are organized and used within the mesh, and it’s quite common in mesh-based simulations. Here’s a breakdown of why `len(mesh.points) > len(unique)` can occur:

# 1. **`mesh.points` Contains All Defined Points**:
#    - `mesh.points` includes **all vertices** defined in the mesh file. These points are indexed and available to be used by any cell in the mesh, even if not all of them are actually needed by the cells that define the object geometry.
#    - Think of `mesh.points` as a master list of all points, some of which might not be connected to any cell in your analysis.

# 2. **Not All Points Are Used in Cells**:
#    - Sometimes, a mesh file includes extra points that aren’t actually part of any cell. This could happen, for example, if the mesh was modified or if parts of the original object were removed, but the points were left in the file.
#    - In such cases, `np.unique(cell_block.data)` would retrieve only the indices of the points actually referenced by cells, ignoring any unused points.

# 3. **Shared Points Between Cells**:
#    - Multiple quadrilateral cells can share vertices. For instance, if two adjacent cells share an edge, the four vertices defining that edge are reused between the two cells.
#    - This reuse of vertices means that when you get the unique indices from all cell data, many vertices will be counted only once, even though they appear multiple times across cells.

# ### Example Scenario

# Consider a simple scenario to make this clearer:

# - Suppose `mesh.points` contains 10 points, but only points 0 through 5 are used to define the cells in the mesh.
# - If you extract unique point indices from the cells (using `np.unique`), you’d end up with only 6 unique points (0 to 5).
# - This would result in `len(mesh.points) = 10`, while `len(unique) = 6`.

# ### Why This Matters

# By using only the unique points associated with cells, the code focuses on points that are actually part of the object’s geometry, filtering out extraneous points that don’t contribute to the final representation. This process improves efficiency, especially in large meshes where unused points or repeated vertices can increase the data size unnecessarily.