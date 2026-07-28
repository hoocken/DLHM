import hydra
from lib.HIT.hit.model.mysmpl import MySmpl
import torch
import numpy as np
import pyvista as pv

from simulator import Simulator

import os
import pickle
import numpy as np
import torch
import trimesh
import shutil

from lib.HIT.hit.utils.model import HitLoader
from lib.HIT.hit.utils.data import load_smpl_data
from lib.HIT.hit.model.deformer import skinning

def predict_occ_from_points(points, hl, data, device):
    device = torch.device(device)
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


    # with open(f'{out_folder}/hit_query_points.pkl', 'wb') as f:
    #     pickle.dump(data_dict, f)

    rgba_colors = class_colors[np.argmax(pred, axis=1)]

    print(rgba_colors.shape)

    point_cloud = trimesh.PointCloud(vertices=points, colors=rgba_colors)
    point_cloud.export("classified_cloud.ply")

    print("Finished querying occupancies and skinning weights!")

    return pred, weights


@hydra.main(version_base=None, config_name='config', config_path='../../config')
def main(config):
    animate_config = config.animate
    target_body = animate_config.target_body
    plot = True if animate_config.plot else False

    bdata = np.load(animate_config.data)

    mesh = pv.read(animate_config.path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load HIT model
    hl = HitLoader.from_expname(animate_config.exp_name, ckpt_choice='best')
    hl.load()
    hl.hit_model.apply_compression = False

    # Create a data dictionary containing the SMPL parameters 
    assert target_body.endswith('.pkl'), 'target_body should be a pkl file'
    assert os.path.exists(target_body), f'SMPL file "{target_body}" does not exist'
    data = load_smpl_data(target_body, device)

    pred, weights = predict_occ_from_points(
        mesh.points,
        hl,
        data,
        device,
    )

    betas = data['betas'].to(device=device, dtype=torch.float).squeeze()

    pred_class = np.argmax(pred, axis=1)

    fps = 60

    output_folder = 'outputs/motion_soft'

    shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    translation = torch.from_numpy(bdata['trans']).to(device=device, dtype=torch.float)
    pose_body = torch.from_numpy(bdata['poses'][:, :66])  # Joint 22 and 23 (hands) are not the same as SMPL
    pose_body = torch.cat((pose_body, torch.zeros(pose_body.shape[0], 6)), dim=1).to(device=device, dtype=torch.float)

    sim = Simulator(
        mesh,
        device,
        pred_class,
        weights,
        young=3e3,
        poisson=0.45,
        damping=0.1,
        dt=0.002,
        plot=plot,
    )

    # Init pose
    output = hl.smpl(betas.unsqueeze(0), translation[0].unsqueeze(0), pose_body[0, 3:].unsqueeze(0),  pose_body[0, :3].unsqueeze(0))
    skinned = skinning(sim.points.to(torch.float32), torch.tensor(sim.mesh.point_data['weights']).to(device), output.tfs, inverse=False)
    sim.init_pose(skinned)

    sim.simulate_one_frame(fps)

    for i in range(1, translation.shape[0]):
        output = hl.smpl(betas.unsqueeze(0), translation[i].unsqueeze(0), pose_body[i, 3:].unsqueeze(0),  pose_body[i, :3].unsqueeze(0))
        skinned = skinning(sim.points.to(torch.float32), torch.tensor(sim.mesh.point_data['weights']).to(device), output.tfs, inverse=False)
        sim.set_pinned_points(skinned)

        sim.simulate_one_frame(fps)

        mesh = sim.fem.mesh.extract_surface()
        mesh.save(f'outputs/motion_soft/frame_{i}.obj')
        print(i)
        
    sim.plotter.close()


if __name__ == "__main__":
    main()