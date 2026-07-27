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

    def combine_mesh(self, mesh1_path, mesh2_path):
        mesh1 = trimesh.load(mesh1_path)
        # Mesh 2: The inner mesh to be subtracted (creating a cavity)
        mesh2 = trimesh.load(mesh2_path)
        target_faces = 5000
        current_faces = len(mesh2.faces)

        if current_faces > target_faces:
            # Calculate the fraction needed to hit the target
            percent_needed = target_faces / current_faces
            downsampled_mesh = mesh2.simplify_quadric_decimation(percent=1 - percent_needed)
        else:
            print("Mesh already has fewer faces than the target.")

        print(f"Mesh 1 (Outer): {len(mesh1.faces)} faces")
        print(f"Mesh 2 (Inner): {len(downsampled_mesh.faces)} faces")

        # Ensure both meshes are clean and watertight for the boolean operation
        mesh1.fill_holes()
        downsampled_mesh.fill_holes()

        print("Combining mesh...")
        combined_mesh = trimesh.util.concatenate([mesh1, downsampled_mesh])
        return combined_mesh

    def create_tetrahedra_mesh(self):
        # mesh = self.combine_mesh(self.smpl_path, self.lean_mesh_path)
        mesh = pv.read(self.smpl_path)

        # Triangulate it to extract raw vertex and face matrices
        tri_mesh = mesh.triangulate()
        vertices = tri_mesh.points

        # Reshape PyVista faces array into a standard (N, 3) matrix
        # PyVista face arrays look like: [3, v0, v1, v2, 3, v0, v1, v2, ...]
        faces = tri_mesh.faces.reshape(-1, 4)[:, 1:]

        tin = pymeshfix.PyTMesh()
        tin.load_array(vertices, faces)
        tin.clean(max_iters=10, inner_loops=3)  # Removes self-intersections & seals mesh

        cleaned_mesh = pv.PolyData(tin.return_points(), np.hstack([np.full((tin.return_faces().shape[0], 1), 3), tin.return_faces()]))



        # Generate the tetrahedral mesh (pytetwild handles the self-intersections)
        print("Fixing self-intersections and generating volume mesh...")
        tgen = tetgen.TetGen(cleaned_mesh)
        # tet_vertices, tet_cells = pytetwild.tetrahedralize(vertices, faces, edge_length_abs=0.2)

        tgen.tetrahedralize(
            mindihedral=10.0,
            minratio=1.5,
        )
        tet_mesh = tgen.grid
        # tet_mesh = pv.UnstructuredGrid({pv.CellType.TETRA: tet_cells}, tet_vertices)

        print(f"Success! Generated {tet_mesh.n_cells} tetrahedra.")
        tet_nodes = tet_mesh.points  # Extract all (N, 3) node coordinates

        # lean_mesh = trimesh.load(self.lean_mesh_path)
        # adipose_mesh = trimesh.load(self.adipose_mesh_path)

        # Ensure the boundary mesh is closed and watertight for accurate queries
        # print("Filling holes for boundary mesh...")
        # lean_mesh.fill_holes()
        # adipose_mesh.fill_holes()
        
        # lean_mask = self._calculate_inside_points(tet_nodes, lean_mesh)
        # tet_mesh.point_data["is_lean"] = lean_mask

        # adipose_mask = self._calculate_inside_points(tet_nodes, adipose_mesh)
        # tet_mesh.point_data["is_adipose"] = adipose_mask

        print("Finished generation of tetrahedralized mesh with marked nodes!")

        self.tet_mesh = tet_mesh
        self.tet_mesh.save(f'{self.out_folder}/tet_mesh.vtu')
        # pcd = pv.PolyData(self.tet_mesh.points)
        # pcd.save(f'{self.out_folder}/tet_mesh.ply', binary=True)

        # tetrahedra = tet_mesh.extract_cells_by_type(pv.CellType.TETRA).cells.reshape(-1, 5)[:, 1:]

        # data_dict = { "points": self.tet_mesh.points, "tetrahedra": tetrahedra}

        # with open(f'{self.out_folder}/tet_mesh.pkl', 'wb') as f:
        #     pickle.dump(data_dict, f)

    def _calculate_inside_points(self, nodes, mesh):
        # Calculate signed distance: negative values are INSIDE, positive are OUTSIDE
        # (A small tolerance ensures nodes exactly on the surface aren't lost)
        print("Calculating inside points...")
        proximity_engine = trimesh.proximity.ProximityQuery(mesh)
        signed_distances = proximity_engine.signed_distance(nodes)
        mask = signed_distances > -1e-3
        return mask

    def plot(self):
        if self.tet_mesh is None:
            print("No mesh to plot!")
            return
        
        print("Starting to plot tetrahedra with marked nodes!")
        
        # Plot the result
        plotter = pv.Plotter()
        plotter.add_mesh(self.tet_mesh, style="wireframe", color="gray", opacity=0.15)

        marked_coordinates = self.tet_mesh.points[self.tet_mesh.point_data["is_lean"]]

        # Build a pure point cloud data structure (contains NO cells or faces)
        marked_points_pcd = pv.PolyData(marked_coordinates)
        plotter.add_mesh(
            marked_points_pcd, 
            color="red", 
            point_size=6, 
            render_points_as_spheres=True, 
            label="Marked Lean Nodes"
        )

        marked_coordinates = self.tet_mesh.points[self.tet_mesh.point_data["is_adipose"]]
        marked_points_pcd = pv.PolyData(marked_coordinates)
        plotter.add_mesh(
            marked_points_pcd, 
            color="yellow", 
            point_size=6, 
            render_points_as_spheres=True, 
            label="Marked Adipose Nodes"
        )

        plotter.add_legend()
        plotter.show()


if __name__ == "__main__":
    tet = Tetrahedralize("outputs/motion/smpl_mesh.obj")
    tet.create_tetrahedra_mesh()
    # tet.plot()