from argparse import ArgumentParser
import pickle

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch

from lib.HIT.hit.model.mysmpl import MySmpl
from lib.SMPL import SMPL 

import subprocess
import re

# TODO: Fix migrating SMPL to MySmpl
def evaluate(points, target):
    target_pcd = o3d.io.read_point_cloud(target)
    target_points = torch.from_numpy(np.asarray(target_pcd.points))

    dist = torch.norm(points - target_points, dim=1)
    mean_error = dist.mean()
    return mean_error * 100

def check_v2v(model: str, id: int, results: list):
    # Load SMPL
    with open('outputs/fit/smpl_fit_params.pkl', 'rb') as f:
        param_dict = pickle.load(f)
    
    trans = param_dict['trans']
    pose = param_dict['pose']
    betas = param_dict['betas']

    smpl = MySmpl('data/models', gender=model)
    
    points = smpl(trans, pose, betas)
    target = f'data/target/tr_reg_{id:03d}.ply'
    v2v = evaluate(points, target)

    print(f"Final V2V to target mesh: {v2v:.4f}")
    results.append(v2v)

if __name__ == "__main__":
    mean = lambda x: sum(x) / len(x)
    # Normal run
    results = []
    # No initialization
    results_no_init = []
    # No pose reg
    results_no_reg = []
    # No shape reg
    results_no_shape_reg = []

    model = ["male", "female", "male", "male", "female", "female", "female", "male", "female", "male"]
    index = 0

    for i in range(0, 100, 2):
        index = i // 10
        print(f"Logging tr_scan_{i:03d}")
        proc = subprocess.Popen(["bash", "run.sh", f"data/input/tr_scan_{i:03d}.ply", model[index], ], stdout=subprocess.PIPE, text=True)
        check_v2v(model, i, results)

        proc = subprocess.Popen(["uv", "run", "fit.py", f"model.base_model={model[index]}", "model.init_epoch=0"], stdout=subprocess.PIPE, text=True)
        check_v2v(model, i, results_no_init)     

        proc = subprocess.Popen(["uv", "run", "fit.py", f"model.base_model={model[index]}", "model.lambda_prior.pose_weight=0", "model.lambda_prior.shape_weight=0"], stdout=subprocess.PIPE, text=True)
        check_v2v(model, i, results_no_reg)   

        # Switch model every 10 scans

    categories = ["Normal", "No Init", "No Reg"]
    values = [mean(results), mean(results_no_init), mean(results_no_reg)]

    # 2. Create the bar graph
    plt.bar(categories, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])

    # 3. Add labels and title
    plt.xlabel("Categories")
    plt.ylabel("V2V")
    plt.title("V2V Error to Registrations")

    plt.savefig("v2v_comparison_graph.png", dpi=300, bbox_inches="tight")

    print(f"Result: {mean(results)} cm")
    print(f"No init: {mean(results_no_init)} cm")
    print(f"No reg: {mean(results_no_reg)} cm")


