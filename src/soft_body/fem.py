import numpy as np
from pyvista import UnstructuredGrid
import pyvista as pv
import torch
from torch_sla import SparseTensor
from tqdm import tqdm

from create_tetrahedral_mesh import Tetrahedralize

class FEM():
    def __init__(self, mesh: UnstructuredGrid, device: torch.device, young: float=1e5, poisson: float=0.3):
        self.mesh = mesh
        self.device = device
        self.points = torch.from_numpy(mesh.points).to(self.device, torch.float64)
        self.X_rest = self.points.clone().detach()
        tet_indices = mesh.extract_cells_by_type(pv.CellType.TETRA).cells.reshape(-1, 5)[:, 1:] # (N, 4)
        self.tets = torch.from_numpy(tet_indices.copy()).to(self.device)

        self.Dm = self._calculate_Dm()
        self.Dm_inv = torch.linalg.inv(self.Dm.transpose(1, 2))

        # Volume of tets
        self.V = torch.abs(torch.linalg.det(self.Dm))/ 6
        self.V = self.V[:, None, None]
        
        self.B = self._calculate_B(self.Dm_inv) 
        self.E = self._calculate_E(young, poisson)

        self.Ke = self.V * (self.B.transpose(1, 2) @ self.E @ self.B)
        self.Ke = self.Ke.to(torch.float64)

        # Mass of each points
        self.M = self._calculate_lumped_mass()
        # Damping factor
        self.c = 0.7

        dof = torch.stack([self.tets * 3, self.tets * 3 + 1, self.tets * 3 + 2], dim=-1) # (N, 4, 3)
        self.dof = dof.reshape(self.tets.shape[0], 12) # (N, 12)

        row = self.dof.unsqueeze(2).expand(-1, -1, 12).reshape(self.tets.shape[0], -1) # (N, 12 * 12)
        col = self.dof.unsqueeze(1).expand(-1, 12, -1).reshape(self.tets.shape[0], -1) # (N, 12 * 12)
        self.row = row.reshape(-1) # (N * 12 * 12)
        self.col = col.reshape(-1) # (N * 12 * 12)

        self.P = self._calculate_P()

        mask = torch.from_numpy(self.mesh.point_data["is_lean"] == False)
        self.mask = mask.unsqueeze(-1).expand(-1, 3).reshape(-1).to(self.device)

        self.collision = 1e-2

    def _lame_parameters(self, E, nu):
        lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return lam, mu
    
    def _calculate_E(self, young, poisson):
        """
        Calculate the matrix E which is the isotropic linear elasticity matrix.
        """
        lam, mu = self._lame_parameters(young, poisson)
        E = torch.zeros((6, 6)).to(self.device)
        E[0:3, 0:3] = lam
        E[0, 0] += 2 * mu
        E[1, 1] += 2 * mu
        E[2, 2] += 2 * mu
        E[3, 3] = mu
        E[4, 4] = mu
        E[5, 5] = mu
        # Expand E to match batch size: (Num_Elements, 6, 6)
        E = E.unsqueeze(0).expand(self.tets.shape[0], -1, -1)
        return E.to(torch.float64)
    
    def _calculate_Dm(self):
        """
        Calculate edge matrix where each row is [x1 - x0, x2 - x0, x3 - x0]
        """
        tet_points = self.points[self.tets]
        Dm = tet_points[:, 1:, :] - tet_points[:, 0:1, :]
        return Dm
    
    def _calculate_B(self, Dm_inv):
        """
        Calculate strain displacement matrix. For a gradient to the i-th node [b_i, c_i, d_i] the matrix B_i is:
        ```
        | b_i 0   0   |
        | 0   c_i 0   |
        | 0   0   d_i |
        | c_i b_i 0   |
        | 0   d_i c_i |
        | d_i 0   b_i |
        ```

        Then the matrices are conactenated horizontally such that we get [ B0, B1, B2, B3 ]
        """
        b_123 = Dm_inv.transpose(1, 2)
        b_0 = - torch.sum(b_123, dim=1, keepdim=True)
        B_spatial = torch.cat([b_0, b_123], dim=1)

        B = torch.zeros((self.tets.shape[0], 6, 12)).to(self.device)
        for i in range(4):
            col = i * 3
            # Direct strain mappings
            B[:, 0, col]     = B_spatial[:, i, 0]
            B[:, 1, col + 1] = B_spatial[:, i, 1]
            B[:, 2, col + 2] = B_spatial[:, i, 2]
            # Shear strain mappings
            B[:, 3, col]     = B_spatial[:, i, 1]
            B[:, 3, col + 1] = B_spatial[:, i, 0]
            
            B[:, 4, col + 1] = B_spatial[:, i, 2]
            B[:, 4, col + 2] = B_spatial[:, i, 1]
            
            B[:, 5, col]     = B_spatial[:, i, 2]
            B[:, 5, col + 2] = B_spatial[:, i, 0]

        return B.to(torch.float64)
    
    def _block_diag_rotation(self, rotation_3x3):
        # rotation_3x3: (N, 3, 3) -> (N, 12, 12) block diagonal, repeated for 4 nodes
        N = rotation_3x3.shape[0]
        R = torch.zeros(N, 12, 12, device=rotation_3x3.device, dtype=torch.float64)
        for i in range(4):
            R[:, i*3:i*3+3, i*3:i*3+3] = rotation_3x3
        return R
    
    def _assemble_Kp(self, rotation):
        """
        Calculate global rotational stiffness matrix
        """
        K_p_local = (rotation @ self.Ke @ rotation.transpose(1, 2))
        # K_p_local = self.Ke
        val = K_p_local.reshape(-1) # (N * 12 * 12)

        K_p = torch.sparse_coo_tensor(
            indices=torch.stack([self.row, self.col]),
            values=val,
            size=(3 * self.points.shape[0], 3 * self.points.shape[0]) # (3N, 3N)
        )
        K_p = K_p.coalesce() # Sum_e K'_e
        return K_p
    
    def _assemble_f0(self, rotation):
        """
        Calculate internal forces
        """
        x_bar_local = self.X_rest[self.tets].reshape(self.tets.shape[0], 12, 1) # (N, 12, 1)
        f0_local = (rotation @ self.Ke @ x_bar_local).squeeze(-1) # (N, 12)
        f0 = torch.zeros(3 * self.X_rest.shape[0]).to(self.device, torch.float64)
        f0 = f0.index_add_(0, self.dof.reshape(-1), f0_local.reshape(-1))
        return f0

    def solve_v_next(self, v, rotation, f_ext, dt):
        rotation = rotation.to(self.device)
        print(rotation)
        rotation = self._block_diag_rotation(rotation)

        K_p = self._assemble_Kp(rotation)
        f0 = self._assemble_f0(rotation)

        left_side = self.M + self.c * dt * torch.ones_like(self.M)

        idx = torch.arange(3 * self.points.shape[0], device=left_side.device).repeat(2, 1)
        left_side = torch.sparse_coo_tensor(idx, left_side, (3 * self.points.shape[0], 3 * self.points.shape[0]))

        left_side = left_side + K_p * dt * dt
        left_side = left_side.coalesce()

        
        f_ext = f_ext.to(self.device)
        v = v.to(self.device)
        right_side = self.M * v - dt * (K_p @ self.points.reshape(-1) - f0 - f_ext)
        
        # Use torch-sla SparseTensor to solve linear eqation
        left_side = SparseTensor(
            values=left_side.values(),
            row_indices=left_side.indices()[0],
            col_indices=left_side.indices()[1],
            shape=left_side.shape,
        )
        v_next = left_side.solve(right_side)
        return v_next * self.mask
    
    def solve_x_next(self, v, dt):
        self.points = self.points + v.reshape(self.points.shape[0], 3) * dt
        return self.points
    
    def _calculate_P(self):
        points = self.points[self.tets] # (N, 4, 3)
        P = torch.zeros(self.tets.shape[0], 4, 4).to(self.device) # (N, 4, 4)
        for i in range(4):
            P[:, i] = torch.hstack([points[:, i], torch.ones(self.tets.shape[0], 1).to(self.device)])

        return P.transpose(1, 2)

    def calculate_R(self):
        Q = self._calculate_P()
        A = torch.linalg.solve(self.P.transpose(1, 2), Q.transpose(1, 2)).transpose(1, 2)
        B = A[:, :3, :3]
        R = self._polar_decomposition(B)
        # print(R)

        # Dm = self._calculate_Dm()
        # R = Dm.transpose(1, 2) @ self.Dm_inv
        # print(R)

        return R

    def _polar_decomposition(self, F):
        """
        Polar decomposition F = R @ S via SVD, batched.
        F: (N, 3, 3) -> R: (N, 3, 3) proper rotation (det = +1)
        """
        U, S, Vh = torch.linalg.svd(F)  # F = U @ diag(S) @ Vh

        # R = U @ Vh gives the closest orthogonal matrix, but may be a
        # reflection (det = -1) if F is inverted/degenerate. Fix by flipping
        # the sign of the smallest singular vector's contribution.
        det = torch.linalg.det(U @ Vh)  # (N,)
        correction = torch.ones_like(S)
        correction[:, -1] = torch.sign(det)  # flip last column if det < 0
        U_corrected = U * correction.unsqueeze(1)  # scale last column of U

        R = U_corrected @ Vh
        return R

    def ground_penalty_force(self, ground_y=0.0, k=50.0):
        depth = torch.clamp(ground_y - self.points[:, 1], min=0.0)  # penetration depth
        f = torch.zeros(3 * self.points.shape[0], device=self.device, dtype=torch.float64)
        f_y_idx = torch.arange(1, 3 * self.points.shape[0], 3, device=self.device)
        f[f_y_idx] = k * depth  # push up proportional to depth
        return f
    
    def self_collision_force(self, k=50.0):
        displacement = self.points[None, :, :] - self.points[:, None, :]
        points_idx = torch.arange(displacement.shape[0])
        
        dist = torch.norm(displacement, p=2, dim=-1)
        dist[points_idx, points_idx] = torch.inf
        min_dist, idx = dist.min(dim=1)

        # How much penetration (negative if there is)
        penetration = torch.clamp(min_dist - self.collision, max=0.0)
        direction = displacement[points_idx, idx, :] / min_dist.unsqueeze(-1).clamp(min=1e-8)  # unit vector, i -> nearest j
        f = k * penetration.unsqueeze(-1) * direction
        f = f.reshape(-1) # (N * 3)
        return f
        
    
    def _calculate_lumped_mass(self, density=1.0):
        tet_mass = density * self.V.squeeze()  # (N,) mass per tet
        node_mass = torch.zeros(self.points.shape[0], device=self.device, dtype=torch.float64)
        node_mass = node_mass.index_add_(0, self.tets.reshape(-1), (tet_mass[:, None] / 4).expand(-1, 4).reshape(-1))
        return node_mass.unsqueeze(-1).expand(-1, 3).reshape(-1)



if __name__ == "__main__":
    # tet = Tetrahedralize("outputs/hit_best/smpl_mesh.obj", "outputs/hit_best/AT_mesh.obj", "outputs/hit_best/LT_mesh.obj")
    # tet.create_tetrahedra_mesh()
    tet_mesh = pv.read("mesh.vtu").scale(10.0, inplace=False)
    fem = FEM(tet_mesh, 'cuda')

    # rot = torch.eye(3).unsqueeze(0).expand(fem.tets.shape[0], -1, -1)
    f_g = torch.zeros(3 * fem.points.shape[0]).to(device='cuda')
    f_g[list(range(1, 3 * fem.points.shape[0], 3))] = -0.1
    v = torch.tensor([0])

    ground_y = fem.points[:, 1].min().item()

    plotter = pv.Plotter()
    mesh = pv.UnstructuredGrid({pv.CellType.TETRA: fem.tets.cpu().numpy()}, fem.points.cpu().numpy())
    actor = plotter.add_mesh(
        pv.PolyData(mesh.points), color="gray", 
        show_edges=True, 
        smooth_shading=True, 
        style="surface", 
        point_size=6, 
        render_points_as_spheres=True, 
    )

    # Ground plane for visual reference
    ground = pv.Plane(
        center=(fem.points[:, 0].mean().item(), ground_y, fem.points[:, 2].mean().item()),
        direction=(0, 1, 0),
        i_size=5, j_size=5,
    )
    plotter.add_mesh(ground, color="lightgray", opacity=0.5)

    plotter.show(interactive_update=True)
    print("Min volume:", fem.V.min().item(), "Max volume:", fem.V.max().item())
    print("Any near-zero:", (fem.V.abs() < 1e-8).sum().item())

    # fem.calculate_R()
    dt = 0.01
    for i in tqdm(range(10000)):
        rot = fem.calculate_R()
        f_ext = torch.tensor([0])
        
        v = fem.solve_v_next(v, rot, f_ext, dt)
        fem.solve_x_next(v, dt)
        mesh = pv.UnstructuredGrid({pv.CellType.TETRA: fem.tets.cpu().numpy()}, fem.points.cpu().numpy())
        surface_mesh = mesh.extract_surface()
        
        # Update mesh points in place (no re-triangulation needed since topology is fixed)
        mesh.points = fem.points.detach().cpu().numpy()
        actor.mapper.SetInputData(pv.PolyData(mesh.points))

        plotter.update()
        # pv.save_meshio(f"outputs/fem/{i}.obj", surface_mesh)
    