import torch
import torch.nn as nn
import torch.nn.functional as F
class BarlowTwinsLoss(torch.nn.Module):

    def __init__(self, device, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param
        self.device = device

#     def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
#         # normalize repr. along the batch dimension
#         z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
#         z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD
#         N = z_a.size(0)
#         D = z_a.size(1)
#         # cross-correlation matrix
#         c = torch.mm(z_a_norm.T, z_b_norm) / N # DxD
 
#         matrix = torch.cat([z_a_norm,z_b_norm],0)
#         _,r,_ = torch.svd(matrix)
#         mini = (r[512:]).pow(2).sum()
#         return -r[511]+ mini
#         # loss
#         c_diff = (c - torch.eye(D,device=self.device)).pow(2) # DxD
#         # multiply off-diagonal elems of c_diff by lambda
#         c_diff[~torch.eye(D, dtype=bool)] *= self.lambda_param
#         loss = c_diff.sum()

#         return loss

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
#         z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0) # NxD
#         z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0) # NxD
        z_a_norm = F.normalize(z_a)
        z_b_norm = F.normalize(z_b)
        z_a_norm_T = F.normalize(z_a.T)
        z_b_norm_T = F.normalize(z_b.T)
        
        N = z_a.size(0)
        D = z_a.size(1)
        # cross-correlation matrix
        twins = torch.mm(z_a_norm_T, z_b_norm_T.T)/D  # DxD
        simclr = torch.mm(z_a_norm, z_b_norm.T)/N  # NxN
        positive = simclr.diag().sum()+twins.diag().sum()
        print(positive)
        twins_neg = (twins-torch.diag_embed(torch.diag(twins))).pow(2).sum()
        simclr_neg = (simclr-torch.diag_embed(torch.diag(simclr))).pow(2).sum()
        print(twins_neg)
        return -positive/10.0+twins_neg+simclr_neg
