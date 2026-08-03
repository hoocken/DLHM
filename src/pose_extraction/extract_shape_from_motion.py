import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm
import trimesh
from lib.HIT.hit.model.deformer import skinning
from lib.HIT.hit.model.mysmpl import MySmpl

if __name__ == "__main__":
    bdata = np.load('data/motion/50020_jiggle_on_toes_poses.npz')
    bdata_dict = dict(bdata)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    betas = torch.from_numpy(bdata_dict['betas']).to(device=device, dtype=torch.float)

    translation = torch.from_numpy(bdata['trans']).to(device=device, dtype=torch.float)
    pose_body = torch.from_numpy(bdata['poses'][:, :66])  # Joint 22 and 23 (hands) are not the same as SMPL
    pose_body = torch.cat((pose_body, torch.zeros(pose_body.shape[0], 6)), dim=1).to(device=device, dtype=torch.float)
    smpl = MySmpl('data/models', 'female').to(device)
    smpl.eval()

    output = smpl(betas[:10].unsqueeze(0), torch.zeros_like(translation[0].unsqueeze(0)), smpl.x_cano().to(betas.device), torch.zeros_like(pose_body[0, :3].unsqueeze(0)))
    
    mesh = trimesh.Trimesh(vertices=output.vertices.squeeze().detach().cpu(), faces=output.faces)
    mesh.export(f'outputs/hit_best/smpl_mesh.obj')

    result = {
        "trans" : torch.zeros_like(translation[0].squeeze()) , # Zero out the translation
        "pose" : torch.zeros_like(pose_body[0].squeeze()), # Zero out the pose
        "betas" : betas[:10].squeeze()
    }

    with open('outputs/fit/smpl_fit_params.pkl', "wb") as f:
        pickle.dump(result, f)
