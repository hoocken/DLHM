from torch import nn
import torch

from .SMPLLoader import SMPLLoader


class SMPL(nn.Module):
    def __init__(self, path, device):
        super(SMPL, self).__init__()
        self.device = device
        self.SMPL_Loader = SMPLLoader(path, device)
        self.pose_shape = self.SMPL_Loader.pose_shape
        self.beta_shape = self.SMPL_Loader.beta_shape
        self.data = self.SMPL_Loader.data
        self.parent = self.SMPL_Loader.parent

    def forward(self, trans, pose, betas):
        """
        Takes trans, pose, and betas to return vertices
        """
        # cal body shapes
        v_shaped = self.data['shapedirs'] @ betas + self.data['v_template'] 
        # cal joint location
        self.J = self.data['J_regressor'] @ v_shaped
        # cal rotation matrix for each joint by rodigues
        self.R = self.rodrigues(pose.reshape((-1, 1, 3)))
        
        R_cube = self.R[1:]
        I_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) + torch.zeros((R_cube.shape[0], 3, 3), dtype=torch.float32)).to(device=self.device)
        lrotmin = (R_cube - I_cube).view(-1)
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.data['posedirs'] @ lrotmin

        vertices = []

        # root joint
        vertices.append(torch.cat((torch.cat((self.R[0], torch.reshape(self.J[0, :], (3, 1))), dim=1), 
                                         torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device)), dim=0))
        # child joint
        for i in range(1, self.data['kintree_table'].shape[1]):
            vertices.append(
                    vertices[self.parent[i]] @ 
                    torch.cat((torch.cat((self.R[i], torch.reshape(self.J[i, :] - self.J[self.parent[i], :], (3, 1))), dim = 1), 
                                         torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=self.device)), dim=0)
                )
                
        stacked = torch.stack(vertices, dim=0)
        tt = stacked @ torch.cat((self.J, torch.zeros((24, 1), dtype=torch.float32, device=self.device)), dim=1).reshape(24, 4, 1)
        vertices = stacked - torch.cat((torch.zeros((tt.shape[0], 4, 3), dtype=torch.float32, device=self.device), tt), dim=2)

        T = torch.tensordot(self.data['weights'], vertices, dims=[[1], [0]])
        rest_shape_h = torch.cat((v_posed, torch.ones(([v_posed.shape[0], 1]), dtype=torch.float32, device=self.device)), dim = 1)
        v = (T @  rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        vertices = v + trans.reshape([1, 3])

        return vertices

    def rodrigues(self, r):        
        theta = torch.norm(r + torch.randn_like(r) * 1e-8, dim=(1, 2), keepdim=True)
        r_hat = r / theta
        cos = torch.cos(theta).to(device=self.device)
        z_stick = torch.zeros(theta.shape[0], dtype=torch.float32).to(device=self.device)
        m = torch.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, - r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3]).to(device=self.device)
        i_cube = (torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) \
             + torch.zeros((theta.shape[0], 3, 3), dtype=torch.float32)).to(device=self.device)
        A = r_hat.permute(0, 2, 1)
        dot = A @ r_hat
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R
    
    def save_obj(self, vertices, fname = './test_smpl.obj'):
        with open( fname, 'w') as fp:
            for v in vertices:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
            # Faces are 1-based, not 0-based in obj files
            for f in self.data['f']+1:
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )
        # todo: save UV texture
        print('Save to ', fname)