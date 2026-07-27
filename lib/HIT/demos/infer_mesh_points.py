"""Given a SMPL parameters, infer the tissues occupancy"""

import argparse
import os
import pickle
import numpy as np
import torch
import trimesh

from hit.utils.model import HitLoader
from hit.utils.data import load_smpl_data
import hit.hit_config as cg
from lib.HIT.hit.model.deformer import skinning
from lib.HIT.hit.utils.tensors import cond_create

def main():
    
    parser = argparse.ArgumentParser(description='Infer tissues from SMPL parameters')
    
    parser.add_argument('--exp_name', type=str, default='rel_male',
                        help='Name of the checkpoint experiment to use for inference' 
                        ) #TODO change to checkpoint path
    parser.add_argument('--target_body', type=str, default='assets/sit.pkl', 
                        help='Path to the SMPL file to infer tissues from')
    parser.add_argument('--mesh_points', type=str, default='assets/sit.pkl', 
                        help='Path to the mesh points to query from')
    parser.add_argument('--out_folder', type=str, default='outputs',
                        help='Output folder to save the generated meshes')
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'],
                        help='Device to use for inference')
    parser.add_argument('--ckpt_choice', type=str, default='best', choices=['best', 'last'],
                        help='Which checkpoint to use for inference')
    parser.add_argument('--betas', help="List of the 2 first SMPL betas to use for inference", nargs='+', type=float, default=[0.0, 0.0])
    # to enter this parameter, use the following syntax: --betas 0.0 0.0
    
    args = parser.parse_args()

    exp_name = args.exp_name
    target_body = args.target_body
    ckpt_choice = args.ckpt_choice
    mesh_points = args.mesh_points
    device = torch.device(args.device)
    
    out_folder = os.path.join(args.out_folder, f'hit_best')
    
    # Create a data dictionary containing the SMPL parameters 
    assert target_body.endswith('.pkl'), 'target_body should be a pkl file'
    assert os.path.exists(target_body), f'SMPL file "{target_body}" does not exist'
    data = load_smpl_data(target_body, device)
    points = trimesh.load(mesh_points).vertices
    print(points)

        
    # Create output folder
    os.makedirs(out_folder, exist_ok=True)
    
    # Load HIT model
    hl = HitLoader.from_expname(exp_name, ckpt_choice=ckpt_choice)
    hl.load()
    hl.hit_model.apply_compression = False

    # Extract the mesh 
    pred, weights = hl.hit_model.forward_points(data['betas'], points, 
                                        body_pose=data['body_pose'], 
                                        global_orient=data['global_orient'], 
                                        transl=data['transl'])

    pred = pred.squeeze()
    class_colors = np.array([
        [0, 0, 0, 255],
        [ 255, 0,  0, 255], # Red: LT
        [ 0, 255, 0, 255],  # Green: AT
        [ 0, 0, 255, 255]   # Blue: BT
    ], dtype=np.uint8)

    print(np.unique(np.argmax(pred, axis=1), return_counts=True))

    rgba_colors = class_colors[np.argmax(pred, axis=1)]
    
    data_dict = {'occ_pred': pred, 'weights': weights}
    with open(f'{out_folder}/hit_query_points.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

    print(rgba_colors.shape)

    point_cloud = trimesh.PointCloud(vertices=points, colors=rgba_colors)
    point_cloud.export("classified_cloud.ply")

    print("Finished saving occupancies and skinning weights!")



if __name__ == '__main__':
    main()