import torch
import torch.nn as nn

class NeRFLoss(torch.nn.modules.loss._Loss):
    def __init__(self, coarse_weight_decay=0.1):
        super(NeRFLoss, self).__init__()
        self.coarse_weight_decay = coarse_weight_decay

    def forward(self, input, target, mask):
        losses = []
        psnrs = []
        for rgb in input:
            mse = (mask * ((rgb - target[..., :3]) ** 2)).sum() / mask.sum()
            losses.append(mse)
            with torch.no_grad():
                psnrs.append(mse_to_psnr(mse))
        losses = torch.stack(losses)
        loss = self.coarse_weight_decay * torch.sum(losses[:-1]) + losses[-1]
        return loss, torch.Tensor(psnrs)

class NormalLoss(torch.nn.modules.loss._Loss):
    def __init__(self, weight_decay=1e-3):
        super(NormalLoss, self).__init__()
        self.weight_decay = weight_decay
    
    def forward(self, input, target, weight):
        loss = self.weight_decay*(weight[...,None]*(target-input)**2).sum()
        return torch.nan_to_num(loss)

def mse_to_psnr(mse):
    return -10.0 * torch.log10(mse)
