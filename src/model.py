"""
Optimize a given model with a 3d point cloud using Chamfer distance
"""
from pathlib import Path
import pickle

import hydra
import open3d as o3d
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from lib.SMPL import SMPL
from .loss import DataLoss, PriorLoss

class Registration(nn.Module):
    def __init__(self, config):
        super(Registration, self).__init__()

        self.config = config 
        self.path = config.base_model
        
        base_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        file_name = Path(config.scan).stem
        self.output_dir = base_dir / file_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.smpl = SMPL(self.path, self.device)
        self.voxel_size = config.voxel_size
        
        self.point_cloud = self._load_ply_as_tensor(config.scan)  

        with open(config.prior, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        self.means = torch.from_numpy(gmm['means'].astype(np.float32)).to(self.device)
        self.covs = torch.from_numpy(gmm['covars'].astype(np.float32)).to(self.device)

        self._initalize_parameters()
        self.to(self.device)

        self.data_loss = DataLoss(config.sigma)
        self.prior_loss = PriorLoss(self.device, self.means, self.covs)
        # self.normal_loss = NormalConsistency()
        self.optimizer = Adam(nn.ParameterList([self.trans, self.pose, self.betas]), lr=config.lr)
        self.scheduler = StepLR(self.optimizer, step_size=config.step_size, gamma=config.decay)


        self.epoch = config.epoch

    def _initalize_parameters(self):
        N = self.means.shape[0]
        self.trans = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.pose = nn.Parameter(torch.zeros(self.smpl.pose_shape, dtype=torch.float32))
        self.betas = nn.Parameter(torch.zeros(self.smpl.beta_shape, dtype=torch.float32))

    def _load_ply_as_tensor(self, path: str):
        point_cloud = o3d.io.read_point_cloud(path)
        downsampled = point_cloud.voxel_down_sample(self.voxel_size)

        # Compute normals if not present
        # downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        #     radius=0.1, max_nn=30))
        
        o3d.io.write_point_cloud(self.output_dir / 'point_cloud.ply', downsampled)
        points = torch.from_numpy(np.asarray(downsampled.points)).to(dtype=torch.float32, device=self.device)
        # normals = torch.from_numpy(np.asarray(downsampled.normals)).to(dtype=torch.float32, device=self.device)
        
        return points
        # return torch.from_numpy(np.asarray(downsampled.points)).to(device=self.device)
    
    def fit(self):
        pbar = tqdm(total=self.epoch, ncols=0, desc="Fit")
        for i in range(self.epoch):
            model = self.smpl(self.trans, self.pose, self.betas)
            # Chamfer distance
            data = self.data_loss(model, self.point_cloud)
            prior = self.prior_loss(self.pose[3:])
            
            # Normal consistency
            # mesh_normals = self.compute_mesh_normals(model)
            # normal = self.normal_loss(model, mesh_normals, 
                                    # self.point_cloud, self.point_normals)
            
            # Combined loss
            total_loss = data + 0.001 * prior
            
            
            self.optimizer.zero_grad()
            
            total_loss.backward()

            self.optimizer.step()
            self.scheduler.step() 

            pbar.update(1)
            pbar.set_postfix(loss=total_loss.item())

        self.smpl.save_obj(self.smpl(self.trans, self.pose, self.betas), fname=self.output_dir / 'smpl_fit.obj')

    # def compute_mesh_normals(self, vertices):
    #     """Compute vertex normals from SMPL mesh"""
    #     faces = self.smpl.data['f'].to(torch.int32)  # Triangle faces
        
    #     # Get vertices of each face
    #     v0 = vertices[faces[:, 0]]
    #     v1 = vertices[faces[:, 1]]
    #     v2 = vertices[faces[:, 2]]
        
    #     # Compute face normals via cross product
    #     face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    #     face_normals = face_normals / (torch.norm(face_normals, dim=1, keepdim=True) + 1e-8)
        
    #     # Average normals per vertex
    #     vertex_normals = torch.zeros_like(vertices)
    #     for i in range(len(faces)):
    #         vertex_normals[faces[i, 0]] += face_normals[i]
    #         vertex_normals[faces[i, 1]] += face_normals[i]
    #         vertex_normals[faces[i, 2]] += face_normals[i]
        
    #     vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-8)
    #     return vertex_normals