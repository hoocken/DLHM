import hydra
from matplotlib import pyplot as plt
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
import time

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

def linspace(start, end, steps):
    t = torch.linspace(0, 1, steps).to(start.device)

    linspaces_2d = start.unsqueeze(1) + t.unsqueeze(0) * (end - start).unsqueeze(1)
    return linspaces_2d.transpose(0, 1)

def get_points_from_obj(file_path):
    """Load obj and rotate 90 degrees along x axis"""
    mesh = trimesh.load(file_path)
    matrix = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])

    mesh.apply_transform(matrix)
    return torch.from_numpy(mesh.vertices)

def line_plot(list_values, title, y_title, labels, colors=['black']):
    x = np.arange(len(list_values[0]))
    for i in range(len(list_values)):
        plt.plot(x, list_values[i], label=labels[i], color=colors[i], linestyle='-', linewidth=2)

    # 3. Add titles and labels
    plt.title(title)
    plt.xlabel("Frames")
    plt.ylabel(y_title)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(f'{title}.png', dpi=300, bbox_inches='tight')
    plt.close()

@hydra.main(version_base=None, config_name='config', config_path='../../config')
def main(config):
    animate_config = config.animate
    target_body = animate_config.target_body
    motion_gt = animate_config.motion_gt
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
    frames_folder = output_folder + '/frames'

    shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(frames_folder, exist_ok=True)

    translation = torch.from_numpy(bdata['trans']).to(device=device, dtype=torch.float)
    pose_body = torch.from_numpy(bdata['poses'][:, :66])  # Joint 22 and 23 (hands) are not the same as SMPL
    pose_body = torch.cat((pose_body, torch.zeros(pose_body.shape[0], 6)), dim=1).to(device=device, dtype=torch.float)

    sim = Simulator(
        mesh,
        device,
        pred_class,
        weights,
        young_fat=5e3,
        young_skin=7e3,
        poisson=0.42,
        damping=0.01,
        dt=0.001,
        plot=plot,
    )

    output = hl.smpl(betas=betas.unsqueeze(0), body_pose=hl.smpl.x_cano().to(betas.device))
    smpl_idx = sim.get_corresponding_smpl(output.vertices)

    # Init pose
    output = hl.smpl(betas.unsqueeze(0), translation[0].unsqueeze(0), pose_body[0, 3:].unsqueeze(0),  pose_body[0, :3].unsqueeze(0))
    skinned = skinning(sim.rest_points.to(torch.float32), torch.tensor(sim.weights).to(device), output.tfs, inverse=False)
    sim.init_pose(skinned)

    # Initial volume
    init_vol, init_vol_template = sim.calculate_volume(skinned)

    # Lists
    dvol_list = []
    dvol_template_list = []
    reconstruction_error_list = []
    reconstruction_error_template_list = []
    vertex_disp_list = []

    time_step = []
    time_frame = []
    
    N = translation.shape[0]
    # N = 3
    for i in range(1, N):
        trans = translation[i - 1]
        pose = pose_body[i - 1]

        trans_next = translation[i]
        pose_next = pose_body[i]

        steps = 16
        
        trans_lin = linspace(trans, trans_next, steps)
        pose_lin = linspace(pose, pose_next, steps)

        start_frame = time.perf_counter()

        for j in range(steps - 1):
            output = hl.smpl(betas.unsqueeze(0), trans_lin[j].unsqueeze(0), pose_lin[j, 3:].unsqueeze(0),  pose_lin[j, :3].unsqueeze(0))
            skinned = skinning(sim.rest_points.to(torch.float32), torch.tensor(sim.weights).to(device), output.tfs, inverse=False)
            sim.set_pinned_points(skinned)
        
            for _ in range(1):
                start_step = time.perf_counter()
                sim.step()
                end_step = time.perf_counter()

                time_step.append(end_step - start_step)

        end_frame = time.perf_counter()
        time_frame.append(end_frame - start_frame)

        # Vertex displacement
        vertex_disp = sim.get_3d_displacements(skinned)
        vertex_disp_list.append(vertex_disp.cpu().numpy())
    
        if sim.plot:
            sim.plot_step()

        # Volume change
        vol, vol_template = sim.calculate_volume(skinned)
        dvol = torch.mean(vol - init_vol)
        dvol_template = torch.mean(vol_template - init_vol_template)
        dvol_list.append(dvol.cpu().numpy())
        dvol_template_list.append(dvol_template.cpu().numpy())

        # Reconstruction error
        gt_points = get_points_from_obj(f"{motion_gt}/{i:05d}.obj")
        points = torch.from_numpy(sim.mesh.points[smpl_idx])
        reconstruction_error = torch.norm(points - gt_points, dim=-1).mean()
        reconstruction_error_list.append(reconstruction_error)

        reconstruction_error_template = torch.norm(output.vertices.cpu() - gt_points, dim=-1).mean()
        reconstruction_error_template_list.append(reconstruction_error_template)

        mesh = sim.mesh.extract_surface()
        mesh.save(frames_folder + f'/frame_{i}.obj')
        print(i)

    for _ in range(10):
        sim.set_pinned_points(skinned)
        sim.simulate_one_frame(fps)

    print(f'Average step time: {sum(time_step) / len(time_step)} seconds')
    print(f'Average frame time: {sum(time_frame) / len(time_frame)} seconds')

    print(f'Average reconstruction error: {sum(reconstruction_error_list) / len(reconstruction_error_list)} m')
    print(f'Average reconstruction error template: {sum(reconstruction_error_template_list) / len(reconstruction_error_template_list)} m')
    print(f'Average vol change: {sum(dvol_list) / len(dvol_list)} m3')
    print(f'Average vol change template: {sum(dvol_template_list) / len(dvol_template_list)} m3')

    avg_disp_list = [disp.mean() for disp in vertex_disp_list]
    print(f'Average vertex displacement: {sum(avg_disp_list) / len(avg_disp_list)} m')

    line_plot([reconstruction_error_list, reconstruction_error_template_list], 'Reconstruction Error', 'Distance (m)', ['Simulated', 'Template'], ['black', 'blue'])
    line_plot([dvol_list, dvol_template_list], 'Volume Change', 'Volume Change', ['Volume Change Simulated', 'Volume Change Template'], ['black', 'blue'])
    line_plot([avg_disp_list], '3D Vertex Displacement', 'Vertex Displacement (m)', ['Vertex Displacement'])

    result = {
        "dvol" : dvol_list,
        "dvol_template": dvol_template_list,
        "reconstruction_error" : reconstruction_error_list,
        "vertex_disp" : vertex_disp_list,
    }

    with open(output_folder + '/measurement.pkl', "wb") as f:
        pickle.dump(result, f)

    print("Finished writing measurements!")

    sim.plotter.close()


if __name__ == "__main__":
    main()