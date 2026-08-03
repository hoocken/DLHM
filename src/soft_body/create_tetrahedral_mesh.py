# Downsample the mesh
# Create tetrahedral mesh from semgentation
# Compose them all 
import os
import pickle

import numpy as np
import pyvista as pv
import trimesh
import pytetwild
import tetgen
import pymeshfix

class Tetrahedralize():
    """
    Create a volumetric tetrahedralized mesh. 
    """
    def __init__(self, smpl_path: str):
        """
        Parameters:
            smpl_path: Path to the obj file of SMPL
            adipose_mesh_path: Path to the calculated adipose tissue mesh
            lean_mesh_path: Path to the calculated lean tissue mesh
        """
        self.smpl_path = smpl_path
        self.tet_mesh: pv.UnstructuredGrid | None = None

        self.out_folder = 'outputs/tet_mesh'

    def create_tetrahedra_mesh(self):
        # mesh = self.combine_mesh(self.smpl_path, self.lean_mesh_path)
        mesh = pv.read(self.smpl_path)

        # Triangulate it to extract raw vertex and face matrices
        tri_mesh = mesh.triangulate()
        vertices = tri_mesh.points

        faces = tri_mesh.faces.reshape(-1, 4)[:, 1:]

        tin = pymeshfix.PyTMesh()
        tin.load_array(vertices, faces)
        tin.clean(max_iters=10, inner_loops=3)  # Removes self-intersections & seals mesh

        cleaned_mesh = pv.PolyData(tin.return_points(), np.hstack([np.full((tin.return_faces().shape[0], 1), 3), tin.return_faces()]))

        print("Fixing self-intersections and generating volume mesh...")
        tgen = tetgen.TetGen(cleaned_mesh)
        # tet_vertices, tet_cells = pytetwild.tetrahedralize(vertices, faces, edge_length_abs=0.2)

        tgen.tetrahedralize(
            mindihedral=10.0,
            minratio=1.5,
        )
        tet_mesh = tgen.grid

        print(f"Success! Generated {tet_mesh.n_cells} tetrahedra.")

        self.tet_mesh = tet_mesh
        self.tet_mesh.save(f'{self.out_folder}/tet_mesh.vtu')

    def plot(self):
        if self.tet_mesh is None:
            print("No mesh to plot!")
            return
        
        # Plot the result
        plotter = pv.Plotter()
        plotter.add_mesh(self.tet_mesh, style="wireframe", color="gray", opacity=0.15)

        plotter.add_legend()
        plotter.show()


if __name__ == "__main__":
    tet = Tetrahedralize("outputs/hit_best/smpl_mesh.obj") # hit_best if using soft tissue
    tet.create_tetrahedra_mesh()
    # tet.plot()