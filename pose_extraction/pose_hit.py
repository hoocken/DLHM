import os
import pickle

import hydra
import numpy as np
import torch
from tqdm import tqdm
import trimesh
from lib.HIT.hit.model.deformer import skinning
from lib.HIT.hit.model.mysmpl import MySmpl
from lib.HIT.hit.utils.model import HitLoader
from lib.SMPL import SMPL
import open3d as o3d

@hydra.main(version_base=None, config_name='config', config_path='../config')
def main(config):
    pose_config = config.pose
    bdata = np.load(pose_config.data)

    with open('outputs/fit/smpl_fit_params.pkl', 'rb') as f:
        data_dict = pickle.load(f)
    

    # Extract the individual pose arrays
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    betas = data_dict['betas'].to(device=device, dtype=torch.float)
    
    # Load pkl
    with open('outputs/hit_best/hit_infer.pkl', 'rb') as f:
        hit_dict = pickle.load(f)
    
    weights = hit_dict['weights']
    meshes = hit_dict['meshes']

    smpl = MySmpl('data/models', 'male').to(device)

    translation = torch.from_numpy(bdata['trans']).to(device=device, dtype=torch.float)
    pose_body = torch.from_numpy(bdata['poses'][:, :66])  # Joint 22 and 23 (hands) are not the same as SMPL
    pose_body = torch.cat((pose_body, torch.zeros(pose_body.shape[0], 6)), dim=1).to(device=device, dtype=torch.float)

    # scene = trimesh.Scene()
    tissues = ['LT', 'AT', 'BT']
    
    for t in tissues:
        os.makedirs(f'outputs/motion_seg/{t}', exist_ok=True)

    for i in tqdm(range(0, 300, 10)):
        output = smpl(betas.unsqueeze(0), translation[i].unsqueeze(0), pose_body[i, 3:].unsqueeze(0),  pose_body[i, :3].unsqueeze(0))
        for c in range(len(tissues)):
            # print(translation[i].shape, pose_body[i].shape, betas.shape)
            vertices = torch.from_numpy(np.asarray(meshes[c].vertices)).to(device=device, dtype=torch.float)
            skinned = skinning(vertices, weights[c].squeeze(), output.tfs, inverse=False)

            mesh = trimesh.Trimesh(vertices=skinned.detach().cpu(), faces=meshes[c].faces)
            mesh.export(f'outputs/motion_seg/{tissues[c]}/motion_{i:04d}.obj')

if __name__ == "__main__":
    main()
        
