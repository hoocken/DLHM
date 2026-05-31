import pickle

import hydra
import numpy as np
import torch

from src import Registration

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

@hydra.main(version_base=None, config_name='config', config_path='config')
def main(config):
    with open(config.model.prior, 'rb') as f:
        gmm = pickle.load(f, encoding='latin1')

    means = torch.from_numpy(gmm['means'].astype(np.float32))
    covs = torch.from_numpy(gmm['covars'].astype(np.float32))

    N = means.shape[0]
    losses = []
    optimizers = [Registration(config.model, means[count], means, covs) for count in range(N)]
    indices = list(range(N))
    epoch_scaling = 3
    # min_loss = -1
    # min_optimizer = None
    # min_count = -1
    start = 0
    while True:
        for count in range(N):
            print(f"Fitting initial pose #{indices[count] + 1}:")
            optimizer = optimizers[count]
            loss = optimizer.fit(start)
            optimizer.epoch *= epoch_scaling

            print(f"Final loss of pose #{indices[count] + 1}: {loss}")
                
            losses.append(loss)
            indices.append(count)

        if N == 1:
            print("Found best model!")
            break

        start = optimizer.epoch // epoch_scaling
        print(f"\nTaking the {N // 2} best models")
        args = argsort(losses)
        optimizers = [optimizers[i] for i in args[:N//2]]
        indices = [indices[i] for i in args[:N//2]]
        losses = []
        N = len(optimizers)

    print(f"Saving SMPL model of minimum loss!")
    optimizers[0].save_smpl()

if __name__ == "__main__":
    main()