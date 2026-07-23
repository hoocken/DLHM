import pyvista as pv
from fem import FEM
import torch

class Simulator():
    def __init__(self, 
                 mesh_path: str, 
                 device: torch.device, 
                 young=1e3, 
                 poisson=0.4, 
                 density=1e3, 
                 damping=0.9,
                 gravity=(0, -9.81, 0),
                 ground_y_offset=None,
                 dt=0.002,
                 plot=False, 
                ):
        self.tet_mesh = pv.read(mesh_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fem =  FEM(
            self.tet_mesh, 
            self.device,
            young=young,
            poisson=poisson,
            density=density,
            damping=damping,
        ) 

        self.f_g = torch.tensor(gravity, dtype=torch.float64, device=device).unsqueeze(0).expand(self.fem.points_count, -1).reshape(3 * self.fem.points_count)
        self.v = torch.zeros(self.fem.points_count * 3, dtype=torch.float64, device=device)

        self.ground_y = None
        if ground_y_offset is not None:
            self.ground_y = ground_y_offset + self.fem.points[:, 1].min().item()

        self.plot = plot
        if plot:
            self.setup_plotter()

        self.dt = dt


    def setup_plotter(self):
        self.plotter = pv.Plotter()
        self.mesh = pv.UnstructuredGrid({pv.CellType.TETRA: self.fem.tets.cpu().numpy()}, self.fem.points.cpu().numpy())
        self.actor = self.plotter.add_mesh(
            self.mesh, color="coral", 
            show_edges=True, 
            smooth_shading=True, 
            style="surface", 
        )

        vel_glyphs = self._make_velocity_glyphs(self.fem.points, self.v, self.fem.points_count)
        f_glyphs = self._make_velocity_glyphs(self.fem.points, self.v, self.fem.points_count)

        self.vel_actor = self.plotter.add_mesh(vel_glyphs, color="yellow")
        self.f_actor = self.plotter.add_mesh(f_glyphs, color="red")

        if self.ground_y is not None:
            ground = pv.Plane(
                        center=(self.fem.points[:, 0].mean().item(), self.ground_y, self.fem.points[:, 2].mean().item()),
                        direction=(0, 1, 0),
                        i_size=50, j_size=50,
                    )
            self.plotter.add_mesh(ground, color="lightgray", opacity=0.5)

        self.plotter.show(interactive_update=True)

    def plot_step(self):
        self.mesh.points = self.fem.points.detach().cpu().numpy()
        self.actor.mapper.SetInputData(self.mesh)

        vel_glyphs = self._make_velocity_glyphs(self.fem.points, self.v, self.fem.points_count, 0.01)
        self.vel_actor.mapper.SetInputData(vel_glyphs)

        f_glyphs = self.make_velocity_glyphs(self.fem.points, self.f, self.fem.points_count, 0.001)
        self.f_actor.mapper.SetInputData(f_glyphs)

        self.plotter.update()

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

        rot = self.fem.calculate_R_shape_matching()

        self.v, self.f = self.fem.solve_v_next(self.v, rot, f_ext, self.dt)

        self.fem.solve_x_next(self.v, self.dt)

    def simulate(self, T=100, frame_rate=30):
        num_substeps = (1 / frame_rate) / self.dt

        assert num_substeps >= 1

        for _ in range(T):
            for _ in range(num_substeps):
                self.step()

            if self.plot:
                self.plot_step()
        