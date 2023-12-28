import numpy as np
from lib.pointops.functions import pointops
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

def block_decider(name):
    if name == 'pointnet2_expand':
        return PointNet2Expand 
    if name == 'simple_with_unit':
        return SimpleBlockWithUnit 
    if name == 'residual_with_unit_before_restoration':
        return ResidualBlockWithUnitBeforeRestoration
    if name == 'downsample':
        return Downsampling 
    if name == 'upsample':
        return Upsampling 
    if name == 'laplacian_unit':
        return LaplacianUnit

class PointNet2Expand(nn.Module):
    def __init__(self, d_in, d_out, config):
        super().__init__()
        self.use_xyz = config.use_xyz
        self.expansion = config.expansion_rate
        d_mid = d_out * self.expansion 
        d_in = d_in + 3 if self.use_xyz else d_in
        self.mlp = nn.Sequential(
            nn.Conv1d(d_in, d_mid, 1),
            nn.BatchNorm1d(d_mid),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_mid, d_out, 1),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, p, pj, x, xj):
        l2 = torch.norm(pj, dim=-1, keepdim=True)
        pj = pj / (torch.max(l2, dim=1, keepdim=True)[0] + 1e-8)
        x = torch.cat([pj, xj], dim=-1) if self.use_xyz else xj # (n, 3+c) or (n, c)
        x = x.permute(0, 2, 1).contiguous()
        x = self.mlp(x) 
        return x.max(2)[0]
        

class SimpleBlockWithUnit(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config, level=None):
        super().__init__()
        func = config.convolution
        self.func = block_decider(func)(d_in, d_out, config)

        self.nsample = nsample_unit = nsample # original setting (dd)

        self.level = level
        if config.unit_name is not None:
            self.unit = block_decider(config.unit_name)(d_out, d_out, nsample_unit, stride, config, level=level)
            self.with_unit = True
        else:
            self.with_unit = False
    
    def forward(self, p, x, o, idx=None, save_path=None):
        N, C = x.size()
        pj_xj, idx = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, return_index=True)  # (m, 3+c, nsample)
        pj, xj = pj_xj[:, :, 0:3], pj_xj[:, :, 3:]
        x = self.func(p, pj, x, xj)
        if self.with_unit:
            p, x, o, idx = self.unit(p, x, o, None, save_path=save_path)
        return p, x, o, idx
 
class ResidualBlockWithUnitBeforeRestoration(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config, level=None):
        super().__init__()
        func = config.convolution
        bottleneck_ratio = config.bottleneck_ratio
        
        self.nsample = nsample_unit = nsample

        self.with_unit = config.with_unit
        d_mid = d_in // bottleneck_ratio
        self.reduction = nn.Sequential(
            nn.Linear(d_in, d_mid),
            nn.BatchNorm1d(d_mid),
            nn.ReLU(inplace=True)
        )
        self.func = block_decider(func)(d_mid, d_mid, config)
        self.unit_op = block_decider(config.unit_name)(d_mid, d_mid, nsample_unit, stride, config, level=level) 
        self.expansion = nn.Sequential(
            nn.Linear(d_mid, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, p, x, o, idx=None, save_path=None):
        N, C = x.size()
        identity = x
        x = self.reduction(x)
        pj_xj, idx = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True, return_index=True)  # (m, 3+c, nsample)
        pj, xj = pj_xj[:, :, 0:3], pj_xj[:, :, 3:]

        x = self.func(p, pj, x, xj) # convolution  

        p, x, o, idx = self.unit_op(p, x, o, None, save_path=save_path)

        x = self.expansion(x)

        x = identity + x

        return p, x, o, idx

class Downsampling(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config, level=None):
        super().__init__()
        self.d_in = d_in
        self.nsample = 16 
        self.stride = stride 
        self.mlp = nn.Sequential(
            nn.Linear(d_in+3, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, p, x, o, idx=None, save_path=None):
        identity = x

        n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
        for i in range(1, o.shape[0]):
            count += (o[i].item() - o[i-1].item()) // self.stride
            n_o.append(count)
        n_o = torch.cuda.IntTensor(n_o)
        idx = pointops.furthestsampling(p, o, n_o)  # (m)
        n_p = p[idx.long(), :]  # (m, 3)
        
        pj_xj, idx = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True, return_index=True)  # (m, 3+c, nsample)
        pj, xj = pj_xj[:, :, :3], pj_xj[:, :, 3:]
        pj = pj / (torch.max(torch.norm(pj, dim=-1, keepdim=True), dim=1, keepdim=True)[0] + 1e-8)
        pj_xj = torch.cat([pj, xj], dim=-1)
        x = self.mlp(pj_xj.max(1)[0])


        return n_p, x, n_o, None

class Upsampling(nn.Module):
    def __init__(self, d_in_sparse_dense, d_out, nsample, stride, config, level=None):
        super().__init__()
        d_in_sparse, d_in_dense = d_in_sparse_dense
        self.nsample = nsample
        self.d_out = d_out
        self.level=level

        self.mlp = nn.Sequential(
            nn.Linear(d_in_sparse+ d_in_dense, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, p1,x1,o1, p2,x2,o2, idx=None, save_path=None):
        '''
            pxo1: dense 
            pxo2: sparse  
        '''
        interpolated = pointops.interpolation(p2, p1, x2, o2, o1)
        x = self.mlp(torch.cat([x1, interpolated], dim=1))
        return p1, x, o1, None 


class LaplacianUnit(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config, level=None):
        super().__init__()
        self.nsample = nsample
        self.level = level
        self.varphi = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.BatchNorm1d(d_in),
            nn.ReLU(inplace=True)
        )   

    def forward(self, p, u, o, idx=None, save_path=None):
        # p, u, o = pxo # (n,3), (n, c), (b)
        N, C = u.size()
        u_t = u
        if idx is None:
            u_n, idx = pointops.queryandgroup(self.nsample, p, p, u, None, o, o, use_xyz=False, return_index=True) # (n,nsample,c)
        else:
            u_n = u[idx, :].view(N, self.nsample, -1)

        Lap = u_n.mean(1) - u
        
        u_tt = self.varphi(Lap) + u_t 

        return p, u_tt, o, idx
