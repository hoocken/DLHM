
import torch
import open3d as o3d

from lib.SMPL import SMPL

smpl = SMPL('data/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl', 'cuda')

trans = torch.zeros(3, dtype=torch.float32)
pose =torch.zeros(smpl.pose_shape, dtype=torch.float32)
betas = torch.zeros(smpl.beta_shape, dtype=torch.float32)
model = smpl(trans, pose, betas)

# Open3D requires float64 or float32 numpy arrays
vertices = model.detach().cpu().numpy().squeeze().astype(np.float64)
# SMPL faces must be converted to an integer numpy array
faces = model.faces.astype(np.int32)

# 2. Create an Open3D TriangleMesh
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(faces)

# 3. Generate the Point Cloud
# Strategy A: Use raw vertices directly as the point cloud
pcd_vertices = o3d.geometry.PointCloud()
pcd_vertices.points = mesh.vertices

# Strategy B: Sample points uniformly across the mesh surface (e.g., 10,000 points)
# This creates a much denser, more even distribution
pcd_sampled = mesh.sample_points_uniformly(number_of_points=10000)

# 4. Save to disk
o3d.io.write_point_cloud("smpl_sampled.ply", pcd_sampled)