import pickle

import numpy as np
import torch
from tqdm import tqdm
import trimesh
from lib.SMPL import SMPL
import open3d as o3d

if __name__ == "__main__":
bdata = np.load('data/motion/Jog_3_poses.npz')

with open('outputs/fit/smpl_fit_params.pkl', 'rb') as f:
    data_dict = pickle.load(f)


# Extract the individual pose arrays
# print(bdata['gender'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
betas = data_dict['betas'].to(device=device, dtype=torch.float)
translation = torch.from_numpy(bdata['trans']).to(device=device, dtype=torch.float)
pose_body = torch.from_numpy(bdata['poses'][:, :66])  # Joint 22 and 23 (hands) are not the same as SMPL
pose_body = torch.cat((pose_body, torch.zeros(pose_body.shape[0], 6)), dim=1).to(device=device, dtype=torch.float)
smpl = SMPL('data/models/SMPL_MALE.pkl', device)
smpl.eval()

# scene = trimesh.Scene()
for i in tqdm(range(0, pose_body.shape[0], 10)):
    # print(translation[i].shape, pose_body[i].shape, betas.shape)
    vertex = smpl(translation[i], pose_body[i], betas)
    # mesh = trimesh.Trimesh(vertices=vertex.cpu().numpy(), faces=smpl.data['f'].cpu().numpy())
    # node_name = f"Frame_{i:03d}"
    smpl.save_obj(vertex, f'outputs/motion/motion_{i:04d}.obj')
    
    # scene.add_geometry(mesh, node_name=node_name, extras={"frame_index": i})

# scene.animations = [{
#     'name': 'SMPL_Vertex_Animation',
#     'time': times.tolist(),
#     'weights': weights.tolist()
# }]

# 6. Export to a single integrated asset
# scene.export("smpl_vertex_animation.glb")
    