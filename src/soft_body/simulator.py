from collections import defaultdict
import pickle
import numpy as np

import pyvista as pv
from pyvista import UnstructuredGrid
from fem import FEM
import torch

class Simulator():
    def __init__(self, 
                 mesh: UnstructuredGrid, 
                 device: torch.device, 
                 tissue_class: np.ndarray,
                 weights: np.ndarray,
                 young_fat=5e3,
                 young_skin=4e4, 
                 poisson=0.43, 
                 density=1e3, 
                 damping=0.9,
                 gravity=(0, 0, -9.81),
                 ground_y_offset=None,
                 dt=0.002,
                 plot=False, 
                ):
        self.mesh = mesh
        self.rest_points = torch.from_numpy(mesh.points).to(device, torch.float64)
        self.tet_indices = mesh.extract_cells_by_type(pv.CellType.TETRA).cells.reshape(-1, 5)[:, 1:]

        # Set point weights
        self.mesh.point_data['weights'] = weights.squeeze().detach().cpu()

        # Get surface
        self.surface = self.mesh.extract_surface()
        surface_cell_ids = np.unique(self.surface.cell_data['vtkOriginalCellIds'])
        self.mesh.cell_data['is_surface'] = False # Mark tets on surface
        self.mesh.cell_data['is_surface'][surface_cell_ids] = True

        # Mark all points with lean tissue
        # self.mesh.point_data['is_lean'] = tissue_class == 1
        self.weights = weights.squeeze()
        self.part_ids = torch.argmax(self.weights, axis=1)
        # Pin hands, feet, and head
        mask = (self.part_ids >= 20) | \
                (self.part_ids == 7) | \
                (self.part_ids == 8) | \
                (self.part_ids == 10) | \
                (self.part_ids == 11) | \
                (self.part_ids == 15)

        self.mesh.point_data['is_lean'] = (tissue_class == 1) | (tissue_class == 3)
        self.mesh.point_data['is_lean'][mask.cpu()] = True

        # Remove all tetrahedra which has all points as lean tissue
        non_lean_tets = np.any(self.mesh.point_data['is_lean'][self.tet_indices] == False, axis=1).nonzero()

        self.extract_mesh = self.mesh.extract_cells(non_lean_tets)
        self.extracted_point_ids = np.unique(self.extract_mesh.point_data['vtkOriginalPointIds'])

        self.rest_mesh = self.mesh.copy(deep=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fem = FEM(
            self.extract_mesh,
            self.device,
            young_fat=young_fat,
            young_skin=young_skin,
            poisson=poisson,
            density=density,
            damping=damping,
        ) 

        self.f_g = torch.tensor(gravity, dtype=torch.float64, device=device).unsqueeze(0).expand(self.fem.points_count, -1).reshape(3 * self.fem.points_count)
        self.f_g = self.f_g * self.fem.M
        self.v = torch.zeros(self.fem.points_count * 3, dtype=torch.float64, device=device)
        self.f = self.f_g

        self.ground_y = None
        if ground_y_offset is not None:
            self.ground_y = ground_y_offset + self.fem.points[:, 1].min().item()

        self.plot = plot
        self.dt = dt

    def init_pose(self, points):
        """Only plot after initializing pose."""
        self.mesh.points = points.cpu().numpy()

        self.fem.points = points[self.extracted_point_ids].to(torch.float64)
        self.mesh.point_data['displacement'] = 0.0
        if self.plot:
            self._setup_plotter()

    def get_corresponding_smpl(self, smpl_points):
        smpl_points = smpl_points.squeeze()
        surface_points = torch.from_numpy(self.surface.points).to(self.device) # (M)

        disp = smpl_points[:, None, :] - surface_points[None, :, :]
        dist = torch.norm(disp, p=2, dim=-1) # (N, M)
        _, min_idx = dist.min(dim=1) # min target for a fixed x; (N)

        orig_id = self.surface.point_data['vtkOriginalPointIds']
        return orig_id[min_idx.cpu()]

    def get_3d_displacements(self, points):
        points = points[self.extracted_point_ids].to(torch.float64)

        mask = ~self.fem.mesh.point_data['is_lean']
        points = points[mask]
        fem_points = self.fem.points[mask]
        disp = torch.norm(fem_points - points, dim=-1)

        orig_id = self.extracted_point_ids[mask]

        self.mesh.point_data['displacement'][orig_id] = disp.cpu() * 1.0

        return disp

    def calculate_volume(self, template_points):
        Dm = self.fem.calculate_Dm(self.fem.points)
        V = self.fem.calculate_V(Dm)

        tet_points = template_points[self.tet_indices]
        Dm_template = tet_points[:, 1:, :] - tet_points[:, 0:1, :]
        V_template = self.fem.calculate_V(Dm_template)
        return V, V_template

    def set_pinned_points(self, points):
        self.mesh.points = points.cpu().numpy()
        old_points = self.fem.points.clone()
        
        points = points.to(torch.float64)
        extracted_points = points[self.extracted_point_ids]
        self.fem.points[self.fem.mesh.point_data['is_lean']] = extracted_points[self.fem.mesh.point_data['is_lean']]
        fixed_dofs = (self.fem.mask).nonzero(as_tuple=True)[0] 

        self.v[fixed_dofs] = points.reshape(-1)[fixed_dofs] - old_points.reshape(-1)[fixed_dofs]

    def _setup_plotter(self):
        self.plotter = pv.Plotter()
        self.plotter.open_gif("mesh_animation.gif")

        self.actor = self.plotter.add_mesh(
            self.mesh, cmap="YlOrRd", 
            clim=[0.0, 0.04],
            scalars="displacement",
            # show_edges=True, 
            smooth_shading=True, 
            style="surface", 
        )

        vel_glyphs = self._make_velocity_glyphs(self.fem.points, self.v, self.fem.points_count)
        f_glyphs = self._make_velocity_glyphs(self.fem.points, self.v, self.fem.points_count)

        self.vel_actor = self.plotter.add_mesh(vel_glyphs, color="yellow")
        self.f_actor = self.plotter.add_mesh(f_glyphs, color="red")

        camera_location = [0.0, -4.0, 2.0]  # Where the camera sits

        mean_center = self.fem.points.mean(dim=0).cpu()
        focal_point = [mean_center[0], mean_center[1], mean_center[2]]      # What the camera looks at
        # view_up = [0.0, 0.0, 1.0] 

        self.plotter.camera.position = camera_location
        self.plotter.camera.focal_point = focal_point

        if self.ground_y is not None:
            ground = pv.Plane(
                        center=(self.fem.points[:, 0].mean().item(), self.ground_y, self.fem.points[:, 2].mean().item()),
                        direction=(0, 1, 0),
                        i_size=50, j_size=50,
                    )
            self.plotter.add_mesh(ground, color="lightgray", opacity=0.5)

        self.plotter.show(interactive_update=True)

    def close_plotter(self):
        self.plotter.close()

    def plot_step(self):
        self.actor.mapper.SetInputData(self.mesh)

        vel_glyphs = self._make_velocity_glyphs(self.fem.points, self.v, self.fem.points_count, 0.0)
        self.vel_actor.mapper.SetInputData(vel_glyphs)

        f_glyphs = self._make_velocity_glyphs(self.fem.points, self.f, self.fem.points_count, 0.000)
        self.f_actor.mapper.SetInputData(f_glyphs)

        self.plotter.update()

        self.plotter.write_frame()

    def _make_velocity_glyphs(self, points, v, points_count, scale=1):
        pts = points.detach().cpu().numpy()
        v3 = v.reshape(points_count, 3).detach().cpu().numpy()
        poly = pv.PolyData(pts)
        poly["velocity"] = v3
        poly.set_active_vectors("velocity")
        # factor controls arrow length relative to vector magnitude; tune to taste
        return poly.glyph(orient="velocity", scale="velocity", factor=scale)

    def step(self):
        f_ext = self.f_g

        if self.ground_y is not None:
            f_ext += self.fem.ground_penalty_force(self.f_g, self.v, self.ground_y, k=1000)

        # f_ext += self.fem.self_collision_force_fast()

        rot = self.fem.calculate_R_shape_matching()

        self.v, self.f = self.fem.solve_v_next(self.v, rot, f_ext, self.dt)

        self.fem.solve_x_next(self.v, self.dt)

        # Update all mesh points
        self.mesh.points[self.extracted_point_ids] = self.fem.points.detach().cpu().numpy()

    def simulate_one_frame(self, frame_rate=30):
        num_substeps = int((1 / frame_rate) // self.dt)

        for _ in range(num_substeps):
            self.step()

        if self.plot:
            self.plot_step()


    def simulate(self, T=1.5, frame_rate=30):
        timesteps = T * frame_rate
        num_substeps = (1 / frame_rate) / self.dt

        assert num_substeps >= 1

        for _ in range(timesteps):
            self.simulate_one_frame(frame_rate)
        