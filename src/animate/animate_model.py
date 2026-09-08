from pathlib import Path
import pickle

import hydra
import numpy as np


@hydra.main(version_base=None, config_name='config', config_path='../../config')
def main(config):
    animate_config = config.animate
    bdata = np.load(animate_config.data)
    bdata_dict = dict(bdata)
    
    with open('outputs/fit/smpl_fit_params.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    bdata_dict['betas'] = data_dict['betas']

    output_path = Path('outputs/motion/motion.npz')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **bdata_dict)
    
if __name__ == "__main__":
    main()