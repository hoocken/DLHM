# Downsample the mesh
# Create tetrahedral mesh from semgentation
# Compose them all 
import pyvista as pv
import trimesh
import pytetwild

class Tetrahedralize():
    """
    Create a volumetric tetrahedralized mesh with marked nodes that is inside the lean and adipose tissue. 
    """
    def __init__(self, smpl_path: str, adipose_mesh_path: str, lean_mesh_path: str):
        """
        Parameters:
            smpl_path: Path to the obj file of SMPL
            adipose_mesh_path: Path to the calculated adipose tissue mesh
            lean_mesh_path: Path to the calculated lean tissue mesh
        """
        self.smpl_path = smpl_path
        self.adipose_mesh_path = adipose_mesh_path
        self.lean_mesh_path = lean_mesh_path
        self.tet_mesh: pv.UnstructuredGrid | None = None

    def create_tetrahedra_mesh(self):
        mesh = pv.read(self.smpl_path)

        # Triangulate it to extract raw vertex and face matrices
        tri_mesh = mesh.triangulate()
        vertices = tri_mesh.points

        # Reshape PyVista faces array into a standard (N, 3) matrix
        # PyVista face arrays look like: [3, v0, v1, v2, 3, v0, v1, v2, ...]
        faces = tri_mesh.faces.reshape(-1, 4)[:, 1:]

        # Generate the tetrahedral mesh (pytetwild handles the self-intersections)
        print("Fixing self-intersections and generating volume mesh...")
        tet_vertices, tet_cells = pytetwild.tetrahedralize(vertices, faces)

        tet_mesh = pv.UnstructuredGrid({pv.CellType.TETRA: tet_cells}, tet_vertices)

        print(f"Success! Generated {tet_mesh.n_cells} tetrahedra.")
        tet_nodes = tet_mesh.points  # Extract all (N, 3) node coordinates

        lean_mesh = trimesh.load(self.lean_mesh_path)
        adipose_mesh = trimesh.load(self.adipose_mesh_path)

        # Ensure the boundary mesh is closed and watertight for accurate queries
        lean_mesh.fill_holes()
        adipose_mesh.fill_holes()
        
        lean_mask = self._calculate_inside_points(tet_nodes, lean_mesh)
        tet_mesh.point_data["is_lean"] = lean_mask

        adipose_mask = self._calculate_inside_points(tet_nodes, adipose_mesh)
        tet_mesh.point_data["is_adipose"] = adipose_mask

        print("Finished generation of tetrahedralized mesh with marked nodes!")

        self.tet_mesh = tet_mesh
        self.tet_mesh.save("mesh.vtu")

    def _calculate_inside_points(self, nodes, mesh):
        # Calculate signed distance: negative values are INSIDE, positive are OUTSIDE
        # (A small tolerance ensures nodes exactly on the surface aren't lost)
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
    tet = Tetrahedralize("outputs/hit_best/smpl_mesh.obj", "outputs/hit_best/AT_mesh.obj", "outputs/hit_best/LT_mesh.obj")
    tet.create_tetrahedra_mesh()
    tet.plot()