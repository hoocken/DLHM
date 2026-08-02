import numpy as np
import torch
from pyvista import UnstructuredGrid
import pyvista as pv
from torch_sla import SparseTensor
from torchtyping import TensorType
from collections import defaultdict
from torch.func import vmap, jacrev, hessian

class FEM():
    def __init__(
            self,
            mesh: UnstructuredGrid, 
            device: torch.device, 
            young_fat: float = 3e3, 
            young_skin: float = 4e4,
            poisson: float = 0.35,
            density: float = 1e3,
            damping: float = 0.0,
        ):
        self.mesh = mesh
        self.device = device

        self.points = torch.from_numpy(mesh.points).to(self.device, torch.float64)
        self.X_rest = self.points.clone().detach()
        tet_indices = mesh.extract_cells_by_type(pv.CellType.TETRA).cells.reshape(-1, 5)[:, 1:] # (N, 4)
        self.tets = torch.from_numpy(tet_indices.copy()).to(self.device)

        self.surface_mask = torch.from_numpy(self.mesh.cell_data['is_surface'])

        self.points_count = self.points.shape[0]
        self.tets_count = self.tets.shape[0]

        self.Dm = self._calculate_Dm(self.X_rest)
        self.V = torch.abs(torch.linalg.det(self.Dm))/ 6
        self.Dm_inv = torch.linalg.inv(self.Dm.transpose(1, 2))

        # Volume of tets
        self.V = self.V[:, None, None]
        
        # Strain-displacement matrix
        self.B = self._calculate_B(self.Dm_inv) 
        # Elasticity matrix
        self.E_fat = self._calculate_E(young_fat, poisson)
        self.E_skin = self._calculate_E(young_skin, poisson)

        self.E = self.E_fat
        self.E[self.surface_mask] = self.E_skin[self.surface_mask]

        # Per-element stiffness matrix
        self.Ke = self.V * (self.B.transpose(1, 2) @ self.E @ self.B)
        self.Ke = self.Ke.to(torch.float64)

        # Mass of each points
        self.M = self._calculate_lumped_mass(density)
        # Damping factor
        self.c = damping

        dof = torch.stack([self.tets * 3, self.tets * 3 + 1, self.tets * 3 + 2], dim=-1) # (M, 4, 3)
        self.dof = dof.reshape(self.tets_count, 12) # (M, 12)

        # Indices from 0 to 3 * N about pairs of nodes (expanded to their x, y, z) that are connected with an edge
        row = self.dof.unsqueeze(2).expand(-1, -1, 12).reshape(self.tets_count, -1) # (M, 12 * 12)
        col = self.dof.unsqueeze(1).expand(-1, 12, -1).reshape(self.tets_count, -1) # (M, 12 * 12)
        self.row = row.reshape(-1) # (M * 12 * 12)
        self.col = col.reshape(-1) # (M * 12 * 12)

        mask = torch.from_numpy(self.mesh.point_data['is_lean'] == True)
        self.mask = mask.unsqueeze(-1).expand(-1, 3).reshape(-1).to(self.device)

        self.weights = torch.from_numpy(self.mesh.point_data['weights']).to(self.device)

    def _lame_parameters(self, E, nu):
        lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        return lam, mu
    
    def _calculate_E(self, young, poisson):
        """
        Calculate the matrix E which is the isotropic linear elasticity matrix.
        """
        lam, mu = self._lame_parameters(young, poisson)
        E = torch.zeros((6, 6)).to(self.device, torch.float64)
        E[0:3, 0:3] = lam
        E[0, 0] += 2 * mu
        E[1, 1] += 2 * mu
        E[2, 2] += 2 * mu
        E[3, 3] = mu
        E[4, 4] = mu
        E[5, 5] = mu
        # Expand E to match batch size: (Num_Elements, 6, 6)
        E = E.unsqueeze(0).expand(self.tets_count, -1, -1)
        return E
    
    def _calculate_Dm(self, points):
        """
        Calculate edge matrix where each row is [x1 - x0, x2 - x0, x3 - x0]
        """
        tet_points = points[self.tets]
        Dm = tet_points[:, 1:, :] - tet_points[:, 0:1, :]
        return Dm.to(torch.float64)
    
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

        B = torch.zeros((self.tets_count, 6, 12)).to(self.device)
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
    
    def _assemble_Kp(self, rotation: TensorType["points", 12, 12]):
        """
        Calculate global rotational stiffness matrix. K' for a node is \sum_e R_e @ K_e @ R_e.T, where
        it is summed over every tetrahedral element with this node.

        Parameters:
            rotation: Tensor of shape (N, 12, 12).
        Returns:
            Kp: Sparse COO Tensor.
        """
        K_p_local = (rotation @ self.Ke @ rotation.transpose(1, 2))
        
        val = K_p_local.reshape(-1) # (N * 12 * 12)

        # K_p = torch.zeros((3 * self.points_count, 3 * self.points_count), dtype=torch.float64, device=self.device)

        # K_p = K_p.index_put_((self.row, self.col), val, accumulate=True)

        K_p = torch.sparse_coo_tensor(
            indices=torch.stack([self.row, self.col]),
            values=val,
            size=(3 * self.points_count, 3 * self.points_count) # (3N, 3N)
        )

        K_p = K_p.coalesce() # Sum_e K'_e
        # K_p = K_p.to_dense()
        return K_p
    
    def _assemble_f0(self, rotation):
        """
        Calculate internal forces
        """
        x_bar_local = self.X_rest[self.tets].reshape(self.tets_count, 12, 1) # (N, 12, 1)
        f0_local = (rotation @ self.Ke @ x_bar_local).squeeze(-1) # (N, 12)
        f0 = torch.zeros(3 * self.X_rest.shape[0]).to(self.device, torch.float64)
        f0 = f0.index_add_(0, self.dof.reshape(-1), f0_local.reshape(-1))
        return f0

    def solve_v_next(self, v: TensorType["points"], rotation, f_ext, dt):
        """
        Calculate v_{t+1} with Euler implicit integration by solving this equation:

        $$
        (M + c * dt + K' * dt^2) v_{t+1} = M * v_t - dt (K @ x_i - f0 - f_ext)
        $$

        Parameters:
            v: Tensor of shape (3 * N), where each element denotes the velocity of a node in x, y, or z axis.
            rotation: Tensor of shape (N, 3, 3)
            f_ext: Tensor of shape (3 * N). External forces.
            dt: float, Timestep
        """
        rotation = rotation.to(self.device)
        rotation = self._block_diag_rotation(rotation)

        K_p = self._assemble_Kp(rotation)
        f0 = self._assemble_f0(rotation)

        # M + c * dt * I
        left_side = self.M + self.c * dt * torch.ones_like(self.M, dtype=torch.float64)
        idx = torch.arange(3 * self.points_count, device=left_side.device).repeat(2, 1)
        left_side = torch.sparse_coo_tensor(idx, left_side, (3 * self.points_count, 3 * self.points_count))

        # M + c * dt * I + Kp * dt * dt
        left_side = left_side + K_p * dt * dt
        left_side = left_side.coalesce()

        f_ext = f_ext.to(self.device)
        v = v.to(self.device)
        forces = K_p @ self.points.reshape(-1) - f0 - f_ext
        right_side = self.M * v - dt * (forces)

        fixed_dofs = (self.mask).nonzero(as_tuple=True)[0] 

        left_side, right_side = self._apply_dirichlet_bc(left_side, right_side, fixed_dofs)
        
        # Use torch-sla SparseTensor to solve linear eqation
        left_side = SparseTensor(
            values=left_side.values(),
            row_indices=left_side.indices()[0],
            col_indices=left_side.indices()[1],
            shape=left_side.shape,
        )
        v_next = left_side.solve(right_side)

        return v_next, forces
    
    def solve_x_next(self, v, dt):
        self.points = self.points + v.reshape(self.points_count, 3) * dt
        return self.points
    
    def _calculate_P(self):
        points = self.points[self.tets] # (N, 4, 3)
        P = torch.zeros(self.tets_count, 4, 4).to(self.device, torch.float64) # (N, 4, 4)
        for i in range(4):
            P[:, i] = torch.hstack([points[:, i], torch.ones(self.tets_count, 1).to(self.device)])

        return P.transpose(1, 2)

    def calculate_R_shape_matching(self, weights=None):
        """
        Per-tet rotation via energy-minimization / shape matching
        (Müller et al. 2005), as used by Georgii & Westermann 2008
        in place of polar-decomposition-of-F for stability under
        large stretch.

        Finds R minimizing sum_i w_i * ||R(X_i - c0) - (x_i - c)||^2
        for each tet's 4 vertices -- i.e. the best rigid alignment of
        the element's rest shape onto its current shape, rather than
        something derived from the deformation gradient.
        """
        tet_points = self.points[self.tets]      # (T,4,3) current
        rest_points = self.X_rest[self.tets]      # (T,4,3) rest

        if weights is None:
            # Uniform per-vertex weight within each element (equal
            # contribution of the 4 corners). Swap in lumped nodal
            # mass here if you want mass-weighted centroids instead.
            w = torch.full((self.tets_count, 4), 0.25,
                            device=self.device, dtype=torch.float64)
        else:
            w = weights  # (T,4)

        c  = (tet_points  * w.unsqueeze(-1)).sum(dim=1)   # (T,3) current centroid
        c0 = (rest_points * w.unsqueeze(-1)).sum(dim=1)   # (T,3) rest centroid

        P = rest_points - c0.unsqueeze(1)   # (T,4,3) centered rest
        Q = tet_points  - c.unsqueeze(1)    # (T,4,3) centered current

        # Cross-covariance H = sum_i w_i * outer(P_i, Q_i)   (T,3,3)
        H = torch.einsum('tv,tva,tvb->tab', w, P, Q)

        U, S, Vh = torch.linalg.svd(H)
        V = Vh.transpose(-2, -1)

        # Reflection correction (Kabsch): flip sign of V's last column
        # if the naive R = V @ U^T would be improper (det = -1).
        det = torch.linalg.det(V @ U.transpose(-2, -1))
        d = torch.ones_like(S)
        d[:, -1] = torch.sign(det)

        R = (V * d.unsqueeze(1)) @ U.transpose(-2, -1)   # (T,3,3)
        return R

    def calculate_R_quaternion(self, weights=None):
        """
        Per-element rotation via energy minimization in quaternion form
        (Georgii & Westermann 2008, Sec. 3 / Muller et al. 2005 / Horn 1987).

        Builds the 4x4 symmetric matrix N whose top eigenvector is the unit
        quaternion q solving the constrained stationarity conditions
        dE/dq + lambda*q = 0, dE/dlambda = 0 from the paper's Eqs. 4-6 --
        i.e. this *is* the solution their Newton solver converges to,
        obtained directly instead of iteratively.
        """
        tet_points = self.points[self.tets]     # (T,4,3) current
        rest_points = self.X_rest[self.tets]    # (T,4,3) rest

        if weights is None:
            w = torch.full((self.tets_count, 4), 0.25,
                            device=self.device, dtype=torch.float64)
        else:
            w = weights  # (T,4)

        c  = (tet_points  * w.unsqueeze(-1)).sum(dim=1)  # current centroid c
        c0 = (rest_points * w.unsqueeze(-1)).sum(dim=1)  # rest centroid c0

        r_prime = tet_points  - c.unsqueeze(1)   # (T,4,3)  x+u-c
        r       = rest_points - c0.unsqueeze(1)  # (T,4,3)  x-c0

        # M[a,b] = sum_i w_i * r'_i[a] * r_i[b]  (current x rest cross-covariance)
        M = torch.einsum('tv,tva,tvb->tab', w, r_prime, r)  # (T,3,3)

        Sxx, Sxy, Sxz = M[:, 0, 0], M[:, 0, 1], M[:, 0, 2]
        Syx, Syy, Syz = M[:, 1, 0], M[:, 1, 1], M[:, 1, 2]
        Szx, Szy, Szz = M[:, 2, 0], M[:, 2, 1], M[:, 2, 2]

        T = self.tets_count
        N = torch.zeros(T, 4, 4, device=self.device, dtype=torch.float64)
        N[:, 0, 0] = Sxx + Syy + Szz
        N[:, 0, 1] = N[:, 1, 0] = Syz - Szy
        N[:, 0, 2] = N[:, 2, 0] = Szx - Sxz
        N[:, 0, 3] = N[:, 3, 0] = Sxy - Syx
        N[:, 1, 1] = Sxx - Syy - Szz
        N[:, 1, 2] = N[:, 2, 1] = Sxy + Syx
        N[:, 1, 3] = N[:, 3, 1] = Szx + Sxz
        N[:, 2, 2] = -Sxx + Syy - Szz
        N[:, 2, 3] = N[:, 3, 2] = Syz + Szy
        N[:, 3, 3] = -Sxx - Syy + Szz

        # N is symmetric -> eigh gives real eigenvalues, ascending order.
        _, eigvecs = torch.linalg.eigh(N)
        q = eigvecs[:, :, -1]   # (T,4) top eigenvector = (w, x, y, z)

        w_, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.zeros(T, 3, 3, device=self.device, dtype=torch.float64)
        R[:, 0, 0] = 1 - 2 * (y*y + z*z)
        R[:, 0, 1] = 2 * (x*y - w_*z)
        R[:, 0, 2] = 2 * (x*z + w_*y)
        R[:, 1, 0] = 2 * (x*y + w_*z)
        R[:, 1, 1] = 1 - 2 * (x*x + z*z)
        R[:, 1, 2] = 2 * (y*z - w_*x)
        R[:, 2, 0] = 2 * (x*z - w_*y)
        R[:, 2, 1] = 2 * (y*z + w_*x)
        R[:, 2, 2] = 1 - 2 * (x*x + y*y)

        return R
    
    def ground_penalty_force(self, gravity: TensorType["points"], v: TensorType["points"], ground_y=0.0, k=5, friction=0.5):
        """
        Calculates penalty force for touching the ground and the friction.

        Parameters:
            gravity (Tensor of shape (N)): Gravity force working on the object.
            v (Tensor of shape (N)): Velocity of each element.
            ground_y (float): y-level of the ground.
            k (int): Multiplicative factor of the penalty force.
            friction (float): Friction coefficient of the ground.

        Returns:
            f (Tensor of shape (N)): Updated force with penalty.
        """
        pen = ground_y - self.points[:, 1]
        gravity = gravity.reshape(-1, 3)
        v = v.reshape(-1, 3).to(self.device)
        depth = torch.clamp(pen, min=0.0)  # penetration depth
        f = torch.zeros(self.points_count, 3, device=self.device, dtype=torch.float64)

        # Push-up force
        f[:, 1] = - gravity[:, 1] * (pen > 0.0) + depth * k # push up proportional to depth

        # Friction
        f[:, 0] = gravity[:, 1] * friction * (pen > 0.0) * torch.sign(v[:, 0])
        f[:, 2] = gravity[:, 1] * friction * (pen > 0.0) * torch.sign(v[:, 2])

        # Re-flatten
        f = f.reshape(-1)
        return f
    
    def _build_point_grid(self, cell_size, max_per_cell=16):
        """
        Bucket self.points into a uniform spatial hash grid, fixed-capacity
        per cell (GPU-friendly counting-sort trick, no Python loop over N).
        Returns (unique_hash, table) where table[b] holds up to max_per_cell
        point indices in the cell with hash unique_hash[b] (-1 = empty slot).
        """
        device = self.points.device
        N = self.points_count
        P1, P2, P3 = 73856093, 19349663, 83492791  # Teschner et al. 2003 hash primes

        def cell_hash(c):
            return (c[..., 0] * P1) ^ (c[..., 1] * P2) ^ (c[..., 2] * P3)

        pc = torch.floor(self.points / cell_size).long()   # (N,3)
        ph = cell_hash(pc)                                   # (N,)

        sort_idx = torch.argsort(ph)
        sorted_hash = ph[sort_idx]
        unique_hash, counts = torch.unique_consecutive(sorted_hash, return_counts=True)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)])

        num_cells = unique_hash.shape[0]
        bucket_id_sorted = torch.repeat_interleave(torch.arange(num_cells, device=device), counts)
        rank_in_bucket = torch.arange(N, device=device) - offsets[bucket_id_sorted]

        table = torch.full((num_cells, max_per_cell), -1, dtype=torch.long, device=device)
        valid = rank_in_bucket < max_per_cell   # drop overflow beyond capacity
        table[bucket_id_sorted[valid], rank_in_bucket[valid]] = sort_idx[valid]

        return unique_hash, table, cell_hash

    def self_collision_force_fast(self, k=500.0, cell_size=None, max_per_cell=16):
        device = self.points.device
        N = self.points_count

        if cell_size is None:
            edge_len = (self.points[self.tets[:, 1:]] - self.points[self.tets[:, 0:1]]).norm(dim=-1)
            cell_size = max(edge_len.max().item() * 1.5, 1e-4)  # comfortably >= one tet's extent

        unique_hash, table, cell_hash = self._build_point_grid(cell_size, max_per_cell)

        tet_pts = self.points[self.tets]           # (T,4,3)
        centroid = tet_pts.mean(dim=1)
        tc = torch.floor(centroid / cell_size).long()   # (T,3)

        offsets_27 = torch.tensor(
            [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
            device=device,
        )

        cand_tets_list, cand_pts_list = [], []
        for off in offsets_27:   # constant 27 iterations -- cheap, not per-N or per-T
            nh = cell_hash(tc + off)                          # (T,)
            idx = torch.searchsorted(unique_hash, nh).clamp(max=unique_hash.shape[0] - 1)
            valid = unique_hash[idx] == nh
            if not valid.any():
                continue
            tet_ids = torch.nonzero(valid, as_tuple=True)[0]
            bucket_pts = table[idx[valid]]                     # (n_valid, max_per_cell)
            tet_ids_exp = tet_ids.unsqueeze(1).expand(-1, max_per_cell).reshape(-1)
            pt_ids_exp = bucket_pts.reshape(-1)
            keep = pt_ids_exp >= 0
            cand_tets_list.append(tet_ids_exp[keep])
            cand_pts_list.append(pt_ids_exp[keep])

        if not cand_tets_list:
            return torch.zeros(3 * N, device=device, dtype=torch.float64)

        cand_tets = torch.cat(cand_tets_list)
        cand_pts = torch.cat(cand_pts_list)

        # A point can't penetrate a tet it's a vertex of
        is_member = (self.tets[cand_tets] == cand_pts.unsqueeze(1)).any(dim=1)
        cand_tets, cand_pts = cand_tets[~is_member], cand_pts[~is_member]
        if cand_tets.numel() == 0:
            return torch.zeros(3 * N, device=device, dtype=torch.float64)

        # ---- narrow phase: same barycentric point-in-tet test as before, candidates only ----
        v0 = tet_pts[cand_tets, 0]
        Dm = (tet_pts[cand_tets, 1:] - v0.unsqueeze(1)).transpose(1, 2)
        ok = Dm.det().abs() > 1e-10
        cand_tets, cand_pts, v0, Dm = cand_tets[ok], cand_pts[ok], v0[ok], Dm[ok]
        if cand_tets.numel() == 0:
            return torch.zeros(3 * N, device=device, dtype=torch.float64)

        Dm_inv = torch.linalg.inv(Dm)
        rel = self.points[cand_pts] - v0
        w123 = torch.einsum('mij,mj->mi', Dm_inv, rel)
        bary = torch.cat([(1.0 - w123.sum(-1)).unsqueeze(-1), w123], dim=-1)
        inside = (bary >= 0.0).all(dim=-1)
        if not inside.any():
            return torch.zeros(3 * N, device=device, dtype=torch.float64)

        cand_tets, cand_pts, bary = cand_tets[inside], cand_pts[inside], bary[inside]
        centroid_in = tet_pts[cand_tets].mean(dim=1)
        out_dir = self.points[cand_pts] - centroid_in
        out_dir = out_dir / out_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        depth = torch.clamp(-bary.min(dim=-1).values, min=0.0)

        f = torch.zeros(N, 3, device=device, dtype=torch.float64)
        f = f.index_add_(0, cand_pts, k * depth.unsqueeze(-1) * out_dir)

        print(f.unique())
        return f.reshape(-1)
    
    def _calculate_lumped_mass(self, density=1000):
        tet_mass = density * self.V[:, 0, 0]  # (N,) mass per tet
        # print(tet_mass)

        node_mass = torch.zeros(self.points_count, device=self.device, dtype=torch.float64)
        node_mass = node_mass.index_put_((self.tets.reshape(-1),), (tet_mass[:, None] / 4).expand(-1, 4).reshape(-1), accumulate=True)
        # return torch.ones(3 * self.points_count,  device=self.device, dtype=torch.float64) * 1e-2
        # return a
        # print("mass", node_mass.unsqueeze(-1).expand(-1, 3).reshape(-1))
        return node_mass.unsqueeze(-1).expand(-1, 3).reshape(-1)

    def _apply_dirichlet_bc(self, K, rhs, fixed_dofs, prescribed_value=0.0):
        """
        Enforce v[fixed_dofs] = prescribed_value by modifying K and rhs in place:
        - zero out rows/cols for fixed DOFs (except diagonal, set to 1)
        - set rhs[fixed_dofs] = prescribed_value
        """
        fixed_set = torch.zeros(K.shape[0], dtype=torch.bool, device=K.device)
        fixed_set[fixed_dofs] = True

        indices = K.indices()
        values = K.values()

        row_fixed = fixed_set[indices[0]]
        col_fixed = fixed_set[indices[1]]

        # Zero any entry where either the row or column is a fixed DOF...
        keep = ~(row_fixed | col_fixed)
        new_values = torch.where(keep, values, torch.zeros_like(values))

        K_bc = torch.sparse_coo_tensor(indices, new_values, K.shape).coalesce()

        # ...then add identity entries back on the diagonal for fixed DOFs
        diag_idx = fixed_dofs
        diag_indices = torch.stack([diag_idx, diag_idx])
        diag_values = torch.ones(diag_idx.shape[0], device=K.device, dtype=K.dtype)

        K_bc = (K_bc + torch.sparse_coo_tensor(diag_indices, diag_values, K.shape)).coalesce()

        rhs_bc = rhs.clone()
        rhs_bc[fixed_dofs] = prescribed_value

        return K_bc, rhs_bc
