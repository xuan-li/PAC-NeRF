
import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']])

total_variation_cuda = load(
        name='total_variation_cuda',
        sources=[
            os.path.join(parent_dir, path)
            for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']])

''' Dense 3D grid
'''
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))

    def forward(self, xyz, transpose=True):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        if transpose:
            out = out.reshape(self.channels,-1).T.reshape(*shape,self.channels)
            if self.channels == 1:
                out = out.squeeze(-1)
        else:
            out = out.reshape(self.channels, *shape)
        return out

    def scale_volume_grid(self, new_world_size, grid_xyz, xyz_min, xyz_max):
        new_grid_value = self.forward(grid_xyz, transpose=False)
        self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]).to(new_grid_value.device))
        if self.channels <= 1:
            self.grid.data[0, 0] = new_grid_value
        else:
            self.grid.data[0] = new_grid_value
        self.xyz_min.data = xyz_min
        self.xyz_max.data = xyz_max

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_cuda.total_variation_add_grad(
            self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'


''' Mask grid
It supports query for the known free space and unknown space.
'''
class MaskGrid(nn.Module):
    def __init__(self, mask=None, xyz_min=None, xyz_max=None):
        super(MaskGrid, self).__init__()
        mask = mask.bool()
        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.tensor(list(mask.shape), device=mask.device) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return f'mask.shape=list(self.mask.shape)'