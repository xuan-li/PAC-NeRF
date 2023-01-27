import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grid import DenseGrid, MaskGrid, render_utils_cuda
from torch_scatter import segment_coo

def interpolation(grid, xyz, xyz_min, xyz_max):
    shape = xyz.shape[:-1]
    xyz = xyz.reshape(1,1,1,-1,3)
    ind_norm = ((xyz - xyz_min) / (xyz_max - xyz_min)).flip((-1,)) * 2 - 1 # N, D, H, W, 3
    out = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=True)
    out = out.reshape(grid.shape[1],-1).T.reshape(*shape, grid.shape[1])
    if grid.shape[1] == 1:
        out = out.squeeze(-1)
    return out

'''Model'''
class DirectVoxGO(torch.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 dx,
                 alpha_init=None,
                 fast_color_thres=0,
                 rgbnet_dim=12, rgbnet_direct=False, 
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4,
                 mask_thres=1e-7,
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        xyz_min = torch.tensor(xyz_min).float()
        xyz_max = torch.tensor(xyz_max).float()
        self.register_buffer('xyz_min_init', xyz_min)
        self.register_buffer('xyz_max_init', xyz_max)
        self.fast_color_thres = fast_color_thres
        self.mask_thres = mask_thres
        # determine the density bias shift
        self.alpha_init = alpha_init
        self.register_buffer("voxel_size_base", torch.FloatTensor([dx]))
        self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(dx)

        self.density = DenseGrid(1, self.world_size, xyz_min, xyz_max) # initial frame
        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
       
        # feature voxel grid + shallow MLP  (fine stage)
        self.k0_dim = rgbnet_dim
        self.k0 = DenseGrid(self.k0_dim, self.world_size, xyz_min, xyz_max)
        self.rgbnet_direct = rgbnet_direct
        self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
        dim0 = (3+3*viewbase_pe*2)
        if rgbnet_direct:
            dim0 += self.k0_dim
        else:
            dim0 += self.k0_dim-3
        self.rgbnet = nn.Sequential(
            nn.Linear(dim0, rgbnet_width), nn.ReLU( inplace=True),
            *[
                nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                for _ in range(rgbnet_depth-2)
            ],
            nn.Linear(rgbnet_width, 3),
        )
        nn.init.constant_(self.rgbnet[-1].bias, 0)
        print('dvgo: feature voxel grid', self.k0.grid.shape)
        print('dvgo: mlp', self.rgbnet)
        
        mask = torch.ones(list(self.world_size), dtype=torch.bool)
        self.mask_cache = MaskGrid(mask=mask,
            xyz_min=xyz_min, xyz_max=xyz_max)

    def _set_grid_resolution(self, dx):
        # Determine grid resolution
        self.voxel_size = dx
        self.world_size = ((self.xyz_max_init - self.xyz_min_init) / self.voxel_size).round().long() + 1

    def get_num_voxel(self):
        return self.world_size.prod()
        
    def get_kwargs(self):
        return {
            'xyz_min': self.xyz_min_init.cpu().numpy(),
            'xyz_max': self.xyz_max_init.cpu().numpy(),
            'alpha_init': self.alpha_init,
            'fast_color_thres': self.fast_color_thres,
            'dx': self.voxel_size,
            'mask_thres': self.mask_thres,
            **self.rgbnet_kwargs,
        }
    
    def generate_particles(self, random=True, ray_o=None, thres=None):
        with torch.no_grad():
            assert not self.density.grid.abs().max().isnan()
            half_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(0, self.world_size[0]-0.5, 2 * (self.world_size[0]-1)),
                torch.linspace(0, self.world_size[1]-0.5, 2 * (self.world_size[1]-1)),
                torch.linspace(0, self.world_size[2]-0.5, 2 * (self.world_size[2]-1)),
            ), -1).to(self.xyz_min_init.device) * self.voxel_size + self.xyz_min_init[None, None, None]
            mask = self.mask_cache(half_grid_xyz)
            half_grid_xyz = half_grid_xyz[mask]
            if random:
                delta = torch.rand_like(half_grid_xyz) * self.voxel_size * 0.5
            else:
                torch.manual_seed(123)
                delta = torch.rand_like(half_grid_xyz) * self.voxel_size * 0.5
                # delta = torch.ones_like(half_grid_xyz) * self.voxel_size * 0.25
            particles = half_grid_xyz + delta
        densities = self.density(particles)
        # alpha = self.activate_density(densities)
        alpha = self.activate_density(densities, 1)
        if thres is None:
            thres = self.mask_thres
        mask = alpha >= alpha.max() * thres
        # mask = (densities - densities.min()) >= (densities.max() - densities.min()) * self.mask_thres
        particles = particles[mask].contiguous()
        densities = densities[mask].contiguous()
        features = torch.cat([densities[...,None], self.k0(particles)], axis=-1).contiguous()
        # velocities = self.velocity(particles).contiguous()
        alpha = alpha[mask].contiguous()

        with torch.no_grad():
            if ray_o is not None:
                k0 = self.k0(particles)
                # view-dependent color emission
                if self.rgbnet_direct:
                    k0_view = k0
                else:
                    k0_view = k0[:, 3:]
                    k0_diffuse = k0[:, :3]
                viewdirs = particles - ray_o[None]
                viewdirs /= viewdirs.norm(dim=-1, keepdim=True)
                viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
                viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
                viewdirs_emb = viewdirs_emb.flatten(0,-2)
                rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
                rgb_logit = self.rgbnet(rgb_feat)
                if self.rgbnet_direct:
                    rgb = torch.sigmoid(rgb_logit)
                else:
                    rgb = torch.sigmoid(rgb_logit + k0_diffuse)
            else:
                rgb = None

        return particles, alpha, features, rgb

    @torch.no_grad()
    def scale_volume_grid(self, dx=None):
        print('dvgo: scale_volume_grid start')
        if dx == None:
            dx = self.voxel_size / 2
        elif abs(dx - self.voxel_size) < 1e-6:
            return
        xyz_min, xyz_max = self.update_occupancy_cache()
        self.xyz_min_init.data = xyz_min
        self.xyz_max_init.data = xyz_max
        ori_world_size = self.world_size
        self._set_grid_resolution(dx)
        # self.act_shift.data = torch.FloatTensor([np.log(1/(1-self.alpha_init) ** dx - 1)])
        # print('dvgo: scale_volume_grid change self.act_shift.data to ', self.act_shift)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self.world_size.tolist())

        grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(0, self.world_size[0]-1, self.world_size[0]),
            torch.linspace(0, self.world_size[1]-1, self.world_size[1]),
            torch.linspace(0, self.world_size[2]-1, self.world_size[2]),
        ), -1).to(xyz_min.device) * self.voxel_size + self.xyz_min_init[None, None, None]

        self.density.scale_volume_grid(self.world_size, grid_xyz, xyz_min, xyz_max)
        self.k0.scale_volume_grid(self.world_size, grid_xyz, xyz_min, xyz_max)
        # self.velocity.scale_volume_grid(self.world_size)
        
        mask = torch.ones(list(self.world_size), dtype=torch.bool, device=self.xyz_min_init.device)
        self.mask_cache = MaskGrid(mask=mask,
            xyz_min=self.xyz_min_init, xyz_max=self.xyz_max_init)
        self.mask_cache.to(self.density.grid.device)
        self.update_occupancy_cache()

        print('dvgo: scale_volume_grid finish')

    @torch.no_grad()
    def update_occupancy_cache(self):
        cache_grid_xyz = torch.stack(torch.meshgrid(
            torch.linspace(0, self.world_size[0]-1, self.world_size[0]),
            torch.linspace(0, self.world_size[1]-1, self.world_size[1]),
            torch.linspace(0, self.world_size[2]-1, self.world_size[2]),
        ), -1).to(self.xyz_min_init.device) * self.voxel_size + self.xyz_min_init[None, None, None]
        cache_grid_density = self.density(cache_grid_xyz)[None,None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = F.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)
        valid_xyz = cache_grid_xyz[self.mask_cache.mask]
        xyz_min = valid_xyz.min(dim=0)[0]
        xyz_max = valid_xyz.max(dim=0)[0]
        return xyz_min, xyz_max

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight / self.world_size.prod()
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight / self.world_size.prod()
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, delta=None):
        delta = delta if delta is not None else 0.5
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.act_shift, delta).reshape(shape)

    @torch.no_grad()
    def sample_ray(self, rays_o, rays_d, near, far, stepsize, xyz_min, xyz_max, **kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * self.voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, xyz_min, xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id

    def forward(self, density_grid, feature_grid, xyz_min, xyz_max, rays_o, rays_d, viewdirs, near=0.1, far=6, stepsize=0.5, bg=1, render_depth=False, t=None, **kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'
        ret_dict = {}
        N = len(rays_o)
        # sample points on rays
        ray_pts, ray_id, step_id = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, near=near, far=far, stepsize=stepsize, xyz_min=xyz_min, xyz_max=xyz_max)
        delta = 0.1 * stepsize * (self.voxel_size / self.voxel_size_base.item())

        density = interpolation(density_grid, ray_pts, xyz_min, xyz_max)
        alpha = self.activate_density(density, delta)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        k0 = interpolation(feature_grid, ray_pts, xyz_min, xyz_max)
        # view-dependent color emission
        if self.rgbnet_direct:
            k0_view = k0
        else:
            k0_view = k0[:, 3:]
            k0_diffuse = k0[:, :3]
        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
        viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
        rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
        rgb_logit = self.rgbnet(rgb_feat)
        if self.rgbnet_direct:
            rgb = torch.sigmoid(rgb_logit)
        else:
            rgb = torch.sigmoid(rgb_logit + k0_diffuse)

        # Ray marching
        torch.cuda.synchronize()
        rgb_marched = segment_coo(
                src=(weights.unsqueeze(-1) * rgb),
                index=ray_id,
                out=torch.zeros([N, 3], device=ray_id.device),
                reduce='sum')
        torch.cuda.synchronize()
        rgb_marched += (alphainv_last.unsqueeze(-1) * bg)
        ret_dict.update({
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
        })

        if render_depth:
            with torch.no_grad():
                torch.cuda.synchronize()
                depth = segment_coo(
                        src=(weights * step_id),
                        index=ray_id,
                        out=torch.zeros([N], device=ray_id.device),
                        reduce='sum')
            ret_dict.update({'depth': depth})
        return ret_dict


''' Misc
'''
class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, delta):
        '''
        alpha = 1 - exp(-softplus(density + shift) * delta)
              = 1 - exp(-log(1 + exp(density + shift)) * delta)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-delta))
              = 1 - (1 + exp(density + shift)) ^ (-delta)
        '''
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, delta)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.delta = delta
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        '''
        alpha' = delta * ((1 + exp(density + shift)) ^ (-delta-1)) * exp(density + shift)'
               = delta * ((1 + exp(density + shift)) ^ (-delta-1)) * exp(density + shift)
        '''
        exp = ctx.saved_tensors[0]
        delta = ctx.delta
        return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), delta), None, None

class Raw2Alpha_nonuni(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        exp, alpha = render_utils_cuda.raw2alpha_nonuni(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return render_utils_cuda.raw2alpha_nonuni_backward(exp, grad_back.contiguous(), interval), None, None

class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, ctx.n_rays, grad_weights, grad_last)
        return grad, None, None


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=c2w.device),
        torch.linspace(0, H-1, H, device=c2w.device))
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+torch.rand_like(i)
        j = j+torch.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = torch.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], torch.ones_like(i)], -1)
    else:
        dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d

@torch.no_grad()
def get_rays_of_a_view(H, W, K, c2w, ndc=False, inverse_y=False, flip_x=False, flip_y=False, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@torch.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert len(rgb_tr) == len(train_poses) and len(rgb_tr) == len(Ks) and len(rgb_tr) == len(HW)
    H, W = HW[0]
    K = Ks[0]
    eps_time = time.time()
    rays_o_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    rays_d_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    viewdirs_tr = torch.zeros([len(rgb_tr), H, W, 3], device=rgb_tr.device)
    imsz = [1] * len(rgb_tr)
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        del rays_o, rays_d, viewdirs
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n].copy_(img.flatten(0,1))
        rays_o_tr[top:top+n].copy_(rays_o.flatten(0,1).to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d.flatten(0,1).to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs.flatten(0,1).to(DEVICE))
        imsz.append(n)
        top += n

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@torch.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    rgb_tr = torch.zeros([N,3], device=DEVICE)
    rays_o_tr = torch.zeros_like(rgb_tr)
    rays_d_tr = torch.zeros_like(rgb_tr)
    viewdirs_tr = torch.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        mask = torch.empty(img.shape[:2], device=DEVICE, dtype=torch.bool)
        for i in range(0, img.shape[0], CHUNK):
            mask[i:i+CHUNK] = model.hit_coarse_geo(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
        n = mask.sum()
        rgb_tr[top:top+n].copy_(img[mask])
        rays_o_tr[top:top+n].copy_(rays_o[mask].to(DEVICE))
        rays_d_tr[top:top+n].copy_(rays_d[mask].to(DEVICE))
        viewdirs_tr[top:top+n].copy_(viewdirs[mask].to(DEVICE))
        imsz.append(n)
        top += n

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # torch.randperm on cuda produce incorrect results in my machine
    idx, top = torch.LongTensor(np.random.permutation(N)), 0
    while True:
        if top + BS > N:
            idx, top = torch.LongTensor(np.random.permutation(N)), 0
        yield idx[top:top+BS]
        top += BS