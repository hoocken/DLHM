import os
from pathlib import Path
import pickle

import hydra
import numpy as np
import torch

from model import Registration

@hydra.main(version_base=None, config_name='config', config_path='../../config')
def main(config):
    with open(config.model.prior, 'rb') as f:
        gmm = pickle.load(f, encoding='latin1')
        
    means = torch.from_numpy(gmm['means'].astype(np.float32))
    covs = torch.from_numpy(gmm['covars'].astype(np.float32))
    weights = torch.from_numpy(gmm['weights'].astype(np.float32))

    mean_shape = means.mean(0)
    seg_result = Path(os.getcwd()) / 'outputs/human3d_segs/segmentation.pkl'
    optimizer = Registration(config.model, mean_shape, means, covs, weights, seg_result)
    
    print(f"Start fitting:")
    loss = optimizer.fit()

    print(f"Final loss: {loss}")

    if config.model.save_model:
        print(f"Saving SMPL model!")
        optimizer.save_smpl()

if __name__ == "__main__":
    main()