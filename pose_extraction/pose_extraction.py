import pickle

import numpy as np
import torch
from tqdm import tqdm
import trimesh
from lib.HIT.hit.model.deformer import skinning
from lib.HIT.hit.model.mysmpl import MySmpl
from lib.SMPL import SMPL
import open3d as o3d

if __name__ == "__main__":
    bdata = np.load('data/motion/Jog_3_poses.npz')

    with open('outputs/fit/smpl_fit_params.pkl', 'rb') as f:
        data_dict = pickle.load(f)


    # Extract the individual pose arrays
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    betas = data_dict['betas'].to(device=device, dtype=torch.float)
    translation = torch.from_numpy(bdata['trans']).to(device=device, dtype=torch.float)
    pose_body = torch.from_numpy(bdata['poses'][:, :66])  # Joint 22 and 23 (hands) are not the same as SMPL
    pose_body = torch.cat((pose_body, torch.zeros(pose_body.shape[0], 6)), dim=1).to(device=device, dtype=torch.float)
    smpl = MySmpl('data/models', 'female').to(device)
    smpl.eval()

    # scene = trimesh.Scene()
    for i in tqdm(range(0, 100, 10)):
        # print(translation[i].shape, pose_body[i].shape, betas.shape)
        output = smpl(betas.unsqueeze(0), translation[i].unsqueeze(0), pose_body[i, 3:].unsqueeze(0),  pose_body[i, :3].unsqueeze(0))

        mesh = trimesh.Trimesh(vertices=output.vertices.squeeze().detach().cpu(), faces=output.faces)
        mesh.export(f'outputs/motion/motion_{i:04d}.obj')
        
    