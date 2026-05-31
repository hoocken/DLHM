import pickle

import numpy as np
import torch
import torch.nn as nn

class DataLoss(nn.Module):
    def __init__(self, sigma):
        super(DataLoss, self).__init__()
        self.sigma = sigma
        
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        
        Parameters:
            x: Shape of (N, 3)
            target: Shape of (M, 3)
        """

        # Chamfer Distance as Point to Surface distance
        # A bit sucky but it'll do
        temp = x[:, None, :]
        temp_target = target[None, :, :]
        dist = torch.norm(temp - temp_target, p=2, dim=-1).pow(2) # (N, M)
        min_x = dist.amin(dim=1) # min target for a fixed x - (N)
        min_target = dist.amin(dim=0) # min x for a fixed target

        # Geman-McClure Penalty
        pow_2 = min_target.pow(2)
        p = pow_2 / (self.sigma + pow_2)
        return p.sum()
    
class PriorLoss(nn.Module):
    def __init__(self, device, means, covs):
        """
        Prior loss towards the pose according to ClothCap paper.
        However, this prior is instead a GMM, with only 69-dimensional
        pose means.
        """
        super(PriorLoss, self).__init__()

        with open(prior_path, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        self.device = device
        self.means = means
        self.covs = covs
        self.inv_covs = self.covs.inverse()


    def forward(self, x):
        """
        Calculates the Mahalanobis distance to each mean, and returns
        the minimum.

        Parameters:
            x: Tensor of shape (69, )
        """
        diff = x - self.means # (N, 69)
        m_dist = ((diff[:, None, :]) @ self.inv_covs).squeeze() # (N, 1, 69)
        m_dist = m_dist @ diff.transpose(0, 1) # (N, N)
        return torch.amin(m_dist.diagonal(), axis=0)


# class NormalConsistency(nn.Module):
#     def __init__(self):
#         super(NormalConsistency, self).__init__()
    
#     def forward(self, mesh_vertices, mesh_normals, 
#                 point_cloud, point_normals) -> torch.Tensor:
#         """
#         Align mesh surface normals with point cloud normals.
        
#         Parameters:
#             mesh_vertices: (V, 3) SMPL vertices
#             mesh_normals: (V, 3) SMPL vertex normals
#             point_cloud: (P, 3) scan points
#             point_normals: (P, 3) scan point normals
#         """
#         # For each point, find nearest mesh vertex
#         dist = torch.cdist(point_cloud, mesh_vertices)  # (P, V)
#         nearest_idx = dist.argmin(dim=1)  # (P,)
        
#         # Get mesh normals at nearest vertices
#         nearest_normals = mesh_normals[nearest_idx]  # (P, 3)
        
#         # Cosine distance between normals (0 = aligned, 1 = opposite)
#         cos_sim = (nearest_normals * point_normals).sum(dim=1)  # (P,)
        
#         # Loss: 1 - |cosine similarity|
#         # Use absolute value to penalize both opposite and perpendicular normals
#         loss = (1.0 - torch.abs(cos_sim)).mean()
        
#         return loss