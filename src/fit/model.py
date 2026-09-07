"""
Optimize a given model with a 3d point cloud using Chamfer distance
"""
from pathlib import Path
import pickle

import open3d as o3d
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import trimesh

from lib.HIT.hit.model.mysmpl import MySmpl
from loss import DataLoss, PosePriorLoss, ShapePriorLoss

smpl_to_body_parts = {
    0: 7, # root -> torso
    1: 15, # left hip -> hips
    2: 15, # right hip -> hips
    3: 15, # belly -> hips
    4: 5, # left knee -> left leg
    5: 11, # right knee -> right leg
    6: 7, # torso -> torso
    7: 6, # left foot -> left foot
    8: 8, # right foot -> right foot
    9: 7, # chest -> torso
    # 10: 6, # left toes -> left feet
    # 11: 8, # right toes -> right feet
    12: 4, # neck -> head
    13: 7, # left chest -> torso
    14: 7, # right chest -> torso
    15: 4, # head -> head
    16: 3, # left shoulder -> left arm
    17: 9, # right shoulder -> right arm
    18: 12, # left elbow -> left fore arm
    19: 13, # right elbow -> right fore arm
    20: 10, # left hand -> left hand
    21: 1, # right hand -> right hand
    # 22: 10, # left fingers -> left hand
    # 23: 1, # right fingers -> right hand
}

class Registration(nn.Module):
    def __init__(self, config, initial_pose, means, covs, weights, seg):
        super(Registration, self).__init__()

        self.config = config 
        self.path = config.base_model
        
        self.output_dir = Path('outputs/fit')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.smpl = MySmpl(self.path, config.gender).to(self.device)
        self.voxel_size = config.voxel_size
        
        self.point_cloud, self.scan_centroids, self.name = self._prepare_point_cloud(seg)
        self.point_cloud = self.point_cloud.to(self.device)

        self.means = means.to(self.device)
        self.covs = covs.to(self.device)
        self.weights = weights.to(self.device)

        self._initalize_parameters(initial_pose)
        self.to(self.device)

        self.data_loss = DataLoss(config.sigma)
        self.pose_prior_loss = PosePriorLoss(self.means, self.covs, self.weights)
        self.shape_prior_loss = ShapePriorLoss()

        self.optimizer = Adam(nn.ParameterList([self.trans, self.pose, self.betas]), lr=config.lr)
        self.scheduler = StepLR(self.optimizer, step_size=config.step_size, gamma=config.decay)

        self.step_size = config.step_size
        self.lambda_prior_pose = config.lambda_prior.pose_weight
        self.lambda_prior_shape = config.lambda_prior.shape_weight
        self.lambda_decay_strength  = config.lambda_prior.strength
        self.lambda_decay_patience = config.lambda_prior.patience
        self.lambda_decay_threshold = config.lambda_prior.threshold

        self.init_epoch = config.init_epoch
        self.epoch = config.epoch

    def _initalize_parameters(self, initial_pose):
        self.trans = nn.Parameter(torch.zeros((1, 3), dtype=torch.float32))
        self.pose = nn.Parameter(initial_pose[None, :])
        self.global_orient = nn.Parameter(torch.zeros((1, 3), dtype=torch.float32))
        self.betas = nn.Parameter(torch.zeros((1, self.smpl.nb_betas), dtype=torch.float32))

    def _prepare_point_cloud(self, segmentations: str):
        with open(segmentations, 'rb') as f:
            data = pickle.load(f)

        body_semseg = data['body_semseg']
        points = data['points']

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        T_y_up_to_z_up = np.array([
                    [1., 0., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., -1., 0., 0.],
                    [0., 0., 0., 1.],
                ])
        
        pcd = pcd.transform(T_y_up_to_z_up)    

        points = np.asarray(pcd.points)

        centroids = np.zeros((np.max(body_semseg) + 1, 3))
        
        # Calculate centroids of each segment
        for i in range(centroids.shape[0]):
            centroids[i] = points[body_semseg == i].mean(axis=0)

        # Combine legs
        centroids[11] = (centroids[11] + centroids[2]) / 2
        centroids[2] = np.nan
        centroids[5] = (centroids[5] + centroids[14]) / 2
        centroids[14] = np.nan
        centroids[0] = np.nan

        # Downsample
        downsampled = pcd.voxel_down_sample(self.voxel_size)
        downsampled_points = torch.from_numpy(np.asarray(downsampled.points)).to(dtype=torch.float32, device=self.device)
        
        # Get name
        name = data['name']

        return downsampled_points, torch.tensor(centroids), name
    
    
    def _calculate_model_centroids(self, model):
        centroids = torch.zeros_like(self.scan_centroids)
        
        segmentations = self.smpl.part_ids.detach().clone()

        mapping = self._map_joints_to_body_parts(segmentations)
        # Calculate centroids
        for i in range(centroids.shape[0]):
            centroids[i] = model[mapping == i].mean(dim=0)

        return centroids
    
    def _map_joints_to_body_parts(self, segmentations):
        parents = self.smpl.parent

        new_mapping = torch.zeros_like(segmentations)
        # Fall back to parent is None
        for i in reversed(smpl_to_body_parts.keys()):
            body = smpl_to_body_parts[i]

            # Fall back for nan values
            centroid = self.scan_centroids[body]
            if torch.all(torch.isnan(centroid)):
                segmentations[segmentations == i] = parents[i]
                
        # Map to body parts
        for i in smpl_to_body_parts.keys():
            new_mapping[segmentations == i] = smpl_to_body_parts[i]
        
        return new_mapping
    
    def initialize_pose(self):
        pbar = tqdm(total=self.init_epoch, initial=0, ncols=0, desc="Initializing")
        total_loss = -1
        for i in range(self.init_epoch):
            output = self.smpl(torch.zeros_like(self.betas), self.trans, self.pose, self.global_orient)
            model = output.vertices.squeeze()

            # Calculate model centroids
            model_centroids = self._calculate_model_centroids(model)
        
            # Body part loss
            diff = torch.nan_to_num((model_centroids - self.scan_centroids), nan=0.0)
            part_loss = torch.norm(diff)
            total_loss = part_loss
            
            self.optimizer.zero_grad()
            
            total_loss.backward()

            self.optimizer.step()
            self.scheduler.step() 

            pbar.update(1)
            pbar.set_postfix(loss=total_loss.item())

    def fit(self, start=0):
        self.initialize_pose()

        pbar = tqdm(total=self.epoch, initial=start, ncols=0, desc="Fit")
        total_loss = -1

        min_loss = -1
        patience = 0

        for i in range(start, self.epoch):
            output = self.smpl(self.betas, self.trans, self.pose, self.global_orient)
            model = output.vertices.squeeze()

            # Chamfer distance
            data = self.data_loss(model, self.point_cloud)
            pose_prior = self.pose_prior_loss(self.pose.squeeze())
            shape_prior = self.shape_prior_loss(self.betas.squeeze())
            
            # Combined loss
            total_loss = data + self.lambda_prior_pose * pose_prior +  self.lambda_prior_shape * shape_prior

            if min_loss == -1 or total_loss < min_loss - self.lambda_decay_threshold:
                min_loss = total_loss
                patience = 0
            else:
                patience += 1

            # Lambda decay
            if patience >= self.lambda_decay_patience:
                patience = 0
                self.lambda_prior_pose = self.lambda_prior_pose * self.lambda_decay_strength
                self.lambda_prior_shape = self.lambda_prior_shape * self.lambda_decay_strength
            
            self.optimizer.zero_grad()
            
            total_loss.backward()

            self.optimizer.step()
            self.scheduler.step() 

            pbar.update(1)
            pbar.set_postfix(loss=total_loss.item())

        return total_loss
    
    @torch.no_grad
    def evaluate(self, target):
        target_pcd = o3d.io.read_point_cloud(target)
        target_points = torch.from_numpy(np.asarray(target_pcd.points)).to(self.device)
        
        model = self.smpl(self.trans, self.pose, self.betas)

        dist = torch.norm(model - target_points, dim=1)
        mean_error = dist.mean()
        return mean_error * 100

    def save_smpl(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud.cpu().numpy())

        o3d.io.write_point_cloud(self.output_dir / 'point_cloud.ply', pcd)
        output = self.smpl(self.betas, self.trans, self.pose, self.global_orient)
        mesh = trimesh.Trimesh(vertices=output.vertices.squeeze().detach().cpu(), faces=output.faces)
        mesh.export(self.output_dir / f'smpl_fit_mesh.obj')
        
        result = {
            "trans" : torch.zeros_like(self.trans.squeeze().cpu()) , # Zero out the translation
            "pose" : torch.zeros(self.pose.shape[1] + self.global_orient.shape[1]), # Zero out the pose
            "betas" : self.betas.squeeze().detach().cpu()
        }

        with open(self.output_dir / 'smpl_fit_params.pkl', "wb") as f:
            pickle.dump(result, f)