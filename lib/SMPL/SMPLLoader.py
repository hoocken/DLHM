'''
    Last modified: 2023.08.02
    Todo: 
        Add a Pytorch implementation
        Add UV texture mapping
        Save joint data
        Harmonize data structure specification
    pokerlishao@gmail.com
'''
import pickle
import os
import numpy as np
import scipy
from . import chumpy
import torch
import torch.nn as nn



class SMPLLoader(pickle.Unpickler, nn.Module):
    '''
    For SMPL v1.1.0
        betas           300 (control body shape)
        pose            72 (control body pose)
        shapedirs       6890 * 3 * 300
        posedirs        6890 * 3 * 207
        v_posed         6890 * 3
        J               24 * 3
        kintree_table   2 * 24 (parent's index and index)
        bs_style        lbs
        weights         6890 * 24
    '''
    def __init__(self ,model_path, device=None):
        nn.Module.__init__(self)
        self.model_path = model_path
        self.device = device if device is not None else torch.device('cpu')
        self.load_model()

    def load_model(self):
        with open (self.model_path,'rb') as f:
            super().__init__(f,encoding='latin1')
            self.data = self.load() # pickle.load()

        self.backwards_compatibility_replacements()
        self.trans_shape = [3]
        self.pose_shape = self.data['kintree_table'].shape[1]*3     # 24 * 3
        self.beta_shape = self.data['shapedirs'].shape[-1]          # 300 in v1.1.0 and 10 in v1.0.0
        self.parent = {
            child_id: parent_id
            for child_id, parent_id in zip(self.data['kintree_table'][1],self.data['kintree_table'][0])
        }
        self.trans2torch()
      
    # help pickle to load pkl file
    def find_class(self, module, name):
        if module == 'chumpy.ch':   # fixed chumpy in local
            return getattr(chumpy.ch, name)
        if module == 'scipy.sparse.csc':    # the `scipy.sparse.csc` namespace is deprecated
            return getattr(scipy.sparse, name) 
        return super().find_class(module, name)
        
    def trans2torch(self):
        for s in ['v_template', 'weights', 'posedirs' ,'shapedirs', 'J']:
            self.data[s] = torch.from_numpy(np.array(self.data[s])).type(torch.float32).to(self.device)
        self.data['J_regressor'] = torch.from_numpy(self.data['J_regressor'].todense()).type(torch.float32).to(self.device)
        self.data['f'] = torch.from_numpy(np.array(self.data['f']).astype(np.int32)).type(torch.float32).to(self.device)
        
    def backwards_compatibility_replacements(self):
        dd = self.data
        # replacements
        if 'default_v' in dd:
            dd['v_template'] = dd['default_v']
            del dd['default_v']
        if 'template_v' in dd:
            dd['v_template'] = dd['template_v']
            del dd['template_v']
        if 'joint_regressor' in dd:
            dd['J_regressor'] = dd['joint_regressor']
            del dd['joint_regressor']
        if 'blendshapes' in dd:
            dd['posedirs'] = dd['blendshapes']
            del dd['blendshapes']
        if 'J' not in dd:
            dd['J'] = dd['joints']
            del dd['joints']

        # defaults
        if 'bs_style' not in dd:
            dd['bs_style'] = 'lbs'
    
    # move to GPU
    def to_device(self, device):
        for name, attr in self.__dict__.items():
            if isinstance(attr, torch.Tensor):
                setattr(self, name, attr.to(device))

def speed_test(smpl,device):
    # about 0.66ms per generation in CPU
    # about 0.89ms per generation in GPU
    import time
    T1 = time.perf_counter()
    for i in range(1000):
        # random pose and shape
        trans = torch.zeros(3, dtype=torch.float32).to(device)
        pose = (torch.rand(smpl.pose_shape, dtype=torch.float32).to(device) - 0.5)
        betas = (torch.rand(smpl.beta_shape, dtype=torch.float32).to(device) - 0.5) * 0.6
        smpl.set_params(trans,pose,betas)
        smpl.cal_shape()
    T2 = time.perf_counter()
    print('Time :%sms' % ((T2 - T1)*1000))  



def main():
    # f_path = 'path/to/smpl_pkl_file'
    f_path = '/home/anthony/Projects/SMPL/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.cuda.set_device(0)

    smpl = SMPLLoader(f_path, device)
    smpl.to_device(device)
    
    # random pose and shape
    # trans = torch.zeros(3, dtype=torch.float32)
    # pose = (torch.rand(smpl.pose_shape, dtype=torch.float32) - 0.5)
    # betas = (torch.rand(smpl.beta_shape, dtype=torch.float32) - 0.5) * 0.6
    # smpl.set_params(trans,pose,betas)
    smpl.cal_shape()

    smpl.save_obj('0.obj')

if __name__ == "__main__":
    main()