"""
Optimize a given model with a 3d point cloud using Chamfer distance
"""
import open3d as o3d
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from lib.SMPL import SMPL
from .loss import Chamfer

class Registration(nn.Module):
    def __init__(self, config):
        super(Registration, self).__init__()

        self.config = config 
        self.path = config.base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.smpl = SMPL(self.path, self.device)
        
        self.point_cloud = self._load_ply_as_tensor(config.scan)  

        self._initalize_parameters()

        self.criterion = Chamfer()
        self.optimizer = Adam(nn.ParameterList([self.trans, self.pose, self.betas]), lr=config.lr)
        self.scheduler = StepLR(self.optimizer, step_size=config.step_size, gamma=config.decay)

        self.epoch = config.epoch

    def _initalize_parameters(self):
        self.trans = nn.Parameter(torch.zeros(3, dtype=torch.float32)).to(device=self.device)
        self.pose = nn.Parameter(torch.rand(self.smpl.pose_shape, dtype=torch.float32) - 0.5).to(device=self.device)
        self.betas = nn.Parameter((torch.rand(self.smpl.beta_shape, dtype=torch.float32) - 0.5) * 0.6).to(device=self.device)

    def _load_ply_as_tensor(self, path: str):
        point_cloud = o3d.io.read_point_cloud(path)
        return torch.from_numpy(np.asarray(point_cloud.points)).to(device=self.device)
    
    def fit(self):
        pbar = tqdm(total=self.epoch, ncols=0, desc="Train")
        for i in range(self.epoch):
            model = self.smpl(self.trans, self.pose, self.betas)
            loss = self.criterion(model, self.point_cloud)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            self.scheduler.step() 

            pbar.update(1)
            pbar.set_postfix(loss=loss)

        

