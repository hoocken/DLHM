import pickle
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

class DataLoss(nn.Module):
    def __init__(self, sigma, distance_type: Literal["Symmetric", "Forward", "Reverse"]="Symmetric"):
        """
        Calculates the point to surface distance by calculating the distance to the nearest point.

        Parameters:
            sigma: Sigma for Geman-McClure Penalty
            distance_type:
                - Symmetric: returns both p2s distance from mesh to target and vice versa
                - Forward: returns p2s distance from mesh to target
                - Reverse: returns p2s distance from target to mesh
        """
        super(DataLoss, self).__init__()
        self.distance_type: Literal["Symmetric", "Forward", "Reverse"] = distance_type
        self.sigma = sigma

    def calculate_point_to_surface_distance(self, x, target):
        temp = x[:, None, :]
        temp_target = target[None, :, :]
        dist = torch.norm(temp - temp_target, p=2, dim=-1) # (N, M)
        min_x = dist.amin(dim=1) # min target for a fixed x; (N)
        min_target = dist.amin(dim=0) # min x for a fixed target; (M)
        return min_x, min_target
        
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the point to surface distance by calculating the distance to the nearest point.
        
        Parameters:
            x: Shape of (N, 3)
            target: Shape of (M, 3)
        """

        # Point to Surface distance
        min_x, min_target = self.calculate_point_to_surface_distance(x, target)

        # Geman-McClure Penalty
        pow_2_min_target = min_target.pow(2)
        p_target = pow_2_min_target / (self.sigma + pow_2_min_target)

        pow_2_min_x = min_x.pow(2)
        p_x = pow_2_min_x / (self.sigma + pow_2_min_x)

        if self.distance_type == "Symmetric":
            return p_target.sum() + p_x.sum()
        elif self.distance_type == "Forward":
            return p_x.sum()
        elif self.distance_type == "Reverse":
            return p_target.sum()
            
    
class PosePriorLoss(nn.Module):
    def __init__(self, means, covs, weights):
        """
        Prior loss towards the pose according to ClothCap paper.
        However, this prior is instead a GMM, with only 69-dimensional
        pose means.
        """
        super(PosePriorLoss, self).__init__()

        self.means = means
        self.covs = covs
        self.weights = weights

        # Pooling
        self.pose_mean = means.mean(0)
        self.pose_cov = (self.weights[:, None, None] * self.covs).sum(0) + ((self.means - self.pose_mean).transpose(0, 1) @ (self.weights[:, None] * (self.means - self.pose_mean)))
        
        self.inv_covs = self.covs.inverse()
        self.inv_pose_cov = self.pose_cov.inverse()
        self.inv_pose_cov = (self.inv_pose_cov + self.inv_pose_cov.T) / 2

    def forward(self, x):
        """
        Calculates the Mahalanobis distance to each mean, and returns
        the minimum.

        Parameters:
            x: Tensor of shape (69, )
        """
        # Individual means
        diff = x - self.means # (N, 69)
        m_dist = ((diff[:, None, :]) @ self.inv_covs).squeeze() # (N, 1, 69)
        m_dist = m_dist @ diff.transpose(0, 1) # (N, N)
        return torch.amin(m_dist.diagonal(), axis=0)

        # Merged mean
        # diff = x - self.pose_mean
        # print(x)
        # print(self.pose_mean)
        # raise Exception
        # m_dist = torch.linalg.solve(self.pose_cov, diff)
        # return diff @ m_dist

class ShapePriorLoss(nn.Module):
    def __init__(self):
        """
        Prior loss towards the pose according to ClothCap paper.
        However, this prior is instead a GMM, with only 69-dimensional
        pose means.
        """
        super(ShapePriorLoss, self).__init__()

    def forward(self, x):
        """
        Parameters:
            x: Tensor of shape (69, )
        """
        return torch.norm(x, p=2).pow(2)
    
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