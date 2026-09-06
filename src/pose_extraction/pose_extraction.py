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

    with open('outputs/fit/smpl_fit_params.pkl', 'rb') as f:
        data_dict = pickle.load(f)


    # Extract the individual pose arrays
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    betas = bdata_dict['betas'].to(device=device, dtype=torch.float)
    print(bdata['betas'].shape)
    # bdata_dict['betas'][:10] = betas.cpu()

    translation = torch.from_numpy(bdata['trans']).to(device=device, dtype=torch.float)
    pose_body = torch.from_numpy(bdata['poses'][:, :66])  # Joint 22 and 23 (hands) are not the same as SMPL
    pose_body = torch.cat((pose_body, torch.zeros(pose_body.shape[0], 6)), dim=1).to(device=device, dtype=torch.float)
    smpl = MySmpl('data/models', 'female').to(device)
    smpl.eval()

    output = smpl(betas[:10].unsqueeze(0), torch.zeros_like(translation[0].unsqueeze(0)), smpl.x_cano().to(betas.device), torch.zeros_like(pose_body[0, :3].unsqueeze(0)))
    
    mesh = trimesh.Trimesh(vertices=output.vertices.squeeze().detach().cpu(), faces=output.faces)
    mesh.export(f'outputs/motion/smpl_mesh.obj')

    result = {
        "trans" : torch.zeros_like(translation[0].squeeze()) , # Zero out the translation
        "pose" : torch.zeros_like(pose_body[0].squeeze()), # Zero out the pose
        "betas" : betas[:10].squeeze()
    }

    with open('outputs/motion/smpl_fit_params.pkl', "wb") as f:
        pickle.dump(result, f)

    # scene = trimesh.Scene()
    # for i in tqdm(range(0, 100, 10)):
    #     # print(translation[i].shape, pose_body[i].shape, betas.shape)
        # output = smpl(betas.unsqueeze(0), translation[i].unsqueeze(0), pose_body[i, 3:].unsqueeze(0),  pose_body[i, :3].unsqueeze(0))

        # mesh = trimesh.Trimesh(vertices=output.vertices.squeeze().detach().cpu(), faces=output.faces)
        # mesh.export(f'outputs/motion/motion_{i:04d}.obj')
        
    