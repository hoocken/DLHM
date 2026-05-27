import torch
import torch.nn as nn

class Chamfer(nn.Module):
    def __init__(self):
        super(Chamfer, self).__init__()
        
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        
        Parameters:
            x: Shape of (N, 3)
            target: Shape of (M, 3)
        """

        temp = x[:, None, :]
        temp_target = target[None, :, :]
        dist = torch.norm(temp - temp_target, p=2, dim=-1).pow(2) # (N, M)
        min_x = dist.amin(dim=1) # min target for a fixed x
        min_target = dist.amin(dim=0) # min x for a fixed target
        return min_x.mean() + min_target.mean()