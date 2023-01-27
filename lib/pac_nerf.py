import numpy as np

from lib.engine.mpm_simulator import MPMSimulator
from .engine import DynamicObserver
from .nerf import DirectVoxGO, get_rays_of_a_view, batch_indices_generator
import torch
from torch import nn
from tqdm import tqdm, trange
import imageio
import os
import torch.nn.functional as F
import taichi as ti
from .engine.mesh_io import write_point_cloud

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def constraint(x, bound):
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return y_scale * torch.tanh(x_scale * x) + (bound[0]+ y_scale)

def constraint_inv(y, bound):
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return torch.arctanh((y - (bound[0] + y_scale)) / y_scale) / x_scale


class PACNeRF(torch.nn.Module):
    def __init__(self,
                xyz_min, 
                xyz_max,
                fast_color_thres,
                mask_thres,
                dtype, 
                dt,
                frame_dt,
                dx,
                alpha_init, 
                particle_chunk_size, 
                rgbnet_dim, 
                rgbnet_depth,
                rgbnet_width,
                viewbase_pe,
                BC = [],
                rgbnet_direct=False,
                write_out=False,
                base_dir=None,
                gravity=[0,0,0],
                cuda_chunk_size = 100,
                material=MPMSimulator.elasticity,
                nerf_bs=2**16,
                taichi_cuda_memory=0.5,
                entropy_weight=0,
                volume_weight=0,
                tv_weight=0,
                **kargs):
        super(PACNeRF, self).__init__()
        self.nerf = DirectVoxGO(xyz_min=xyz_min, xyz_max=xyz_max,
                                alpha_init=alpha_init, fast_color_thres=fast_color_thres,
                                rgbnet_dim=rgbnet_dim, rgbnet_direct=rgbnet_direct, 
                                rgbnet_depth=rgbnet_depth, rgbnet_width=rgbnet_width,
                                viewbase_pe=viewbase_pe, 
                                mask_thres=mask_thres,
                                dx=dx, base_dir=base_dir)
        self.nerf_bs = nerf_bs
        ti.reset()
        ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=taichi_cuda_memory)
        self.dynamic_observer = DynamicObserver(dtype=dtype, rgbnet_dim=rgbnet_dim, dt=dt, frame_dt=frame_dt, 
                                 particle_chunk_size=particle_chunk_size, cuda_chunk_size=cuda_chunk_size,
                                 base_dir=base_dir, gravity=gravity, material=material)
        self.observations = []
        self.frame_dt = frame_dt
        self.global_v = nn.Parameter(torch.tensor([0.,0.,0.]))
        
        if 'E' in kargs and 'nu' in kargs:
            self.global_E = nn.Parameter(torch.tensor([np.log10(kargs['E'])]).float())
            self.nu_bound = [-0.45, 0.45]
            self.global_nu = nn.Parameter(constraint_inv(torch.tensor([kargs['nu']]), self.nu_bound))
        elif 'mu' in kargs and 'kappa' in kargs:
            self.global_mu = nn.Parameter(torch.tensor([np.log10(kargs['mu'])]).float())
            self.global_kappa = nn.Parameter(torch.tensor([np.log10(kargs['kappa'])]).float())
        elif 'mu' in kargs and 'lambda' in kargs:
            self.global_mu = nn.Parameter(torch.tensor([np.log10(kargs['mu'])]).float())
            self.global_lam = nn.Parameter(torch.tensor([np.log10(kargs['lam'])]).float())
        
        self.global_rho = nn.Parameter(torch.tensor([3.0]).float())
        if 'rho' in kargs:
            self.global_rho.data = torch.tensor([np.log10(kargs['rho'])]).float()
        self.friction_angle = nn.Parameter(torch.tensor([0.0]))
        if 'friction_angle' in kargs:
            self.friction_angle.data = torch.tensor([kargs['friction_angle']]).float()
        self.cohesion = nn.Parameter(torch.tensor([0.0]))
        
        self.yield_stress = nn.Parameter(torch.tensor([0.0]))
        if 'yield_stress' in kargs:
            self.yield_stress.data = torch.tensor([np.log10(kargs['yield_stress'])]).float()

        self.plastic_viscosity = nn.Parameter(torch.tensor([-1e6]))
        if 'plastic_viscosity' in kargs:
            self.plastic_viscosity.data = torch.tensor([np.log10(kargs['plastic_viscosity'])]).float()
        
        self.plastic_pow = nn.Parameter(torch.tensor([1.0]))
        if 'plastic_pow' in kargs:
            self.plastic_pow.data = torch.tensor([kargs['plastic_pow']]).float()
        
        
        self.register_buffer('epoch', torch.tensor([-1]))
        self.write_out = write_out
        self.init_particles = None
        self.init_velocities = None
        self.init_rhos = None
        self.init_features = None
        self.init_alphas = None
        self.particle_rgb = None
        self.init_lam = None
        self.init_mu = None
        self.init_yield_stress = None
        self.init_friction_alpha = None
        self.init_plastic_viscosity = None
        self.base_dir = base_dir
        self.material = material
        self.entropy_weight = entropy_weight
        self.volume_weight = volume_weight
        self.tv_weight = tv_weight
        self.batch_indices_generator = None

        for bc in BC:
            if "ground" in bc:
                self.dynamic_observer.simulator.add_surface_collider(BC[bc][0], BC[bc][1], MPMSimulator.surface_sticky)
            elif "cylinder" in bc:
                self.dynamic_observer.simulator.add_cylinder_collider(BC[bc][0], BC[bc][1], BC[bc][2], MPMSimulator.surface_sticky)


    def initialize_particles(self, random=True, ray_o=None, thres=None):
        self.nerf.update_occupancy_cache()
        new_particles, alphas, features, rgb = self.nerf.generate_particles(random, ray_o, thres)
        self.particle_rgb = rgb
        self.init_alphas = alphas
        self.init_rhos = alphas * (10. ** self.global_rho)
        self.init_particles = new_particles
        self.init_features = features
        self.init_velocities = self.global_v.repeat([new_particles.shape[0], 1])
        if hasattr(self, 'global_E') and hasattr(self, 'global_nu'):
            E = 10 ** self.global_E
            nu = constraint(self.global_nu, self.nu_bound)
            mu = E / (2. * (1. + nu))
            lam = E * nu / ((1. + nu) * (1. - 2. * nu))
            self.init_mu = (alphas ** 3) * mu
            self.init_lam = (alphas ** 3) * lam
        if hasattr(self, 'global_mu'):
            self.init_mu = (alphas ** 3) * 10 ** self.global_mu
        if hasattr(self, 'global_lam'):
            self.init_lam = (alphas ** 3) * 10 ** self.global_lam
        if hasattr(self, 'global_kappa'):
            lam = 10 ** self.global_kappa - 2./3. * 10 ** self.global_mu
            self.init_lam = (alphas ** 3) * lam
        self.init_yield_stress = 10 ** self.yield_stress
        self.init_plastic_viscosity = 10 ** self.plastic_viscosity
        sin_phi = torch.sin(self.friction_angle / 180 * np.pi)
        friction_alpha = np.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)
        self.init_friction_alpha = friction_alpha
        
        with torch.no_grad():
            info_str = f"velocity: {self.global_v.tolist()}; rho: {10 ** self.global_rho.item()}"
            if hasattr(self, 'global_E'):
                info_str += f"; E: {10**self.global_E.item()}"
            if hasattr(self, "global_nu"):
                info_str += f"; nu: {constraint(self.global_nu, self.nu_bound).item()}"
            if hasattr(self, "global_mu"):
                info_str += f"; mu: {10 ** self.global_mu.item()}"
            if hasattr(self, "global_lam"):
                info_str += f"; mu: {10 ** self.global_lam.item()}"
            if hasattr(self, "global_kappa"):
                info_str += f"; kappa: {10 ** self.global_kappa.item()}"
            if self.material == MPMSimulator.drucker_prager:
                info_str += f"; friction_angle: {self.friction_angle.item()}"
            elif self.material == MPMSimulator.von_mises:
                info_str += f"; yield_stress: {self.init_yield_stress.item()}; eta: {self.init_plastic_viscosity.item()}"
            print(info_str)

    def reset(self):
        self.observations.clear()

    def get_kwargs(self):
        kwargs = {
            **self.nerf.get_kwargs(),
            **self.dynamic_observer.get_kwargs()
        }
        return kwargs

    def forward(self, max_f, rays_o, rays_d, viewdirs, target, rays_mask, bg, H, W, saving_freq=100, partial=False, full_batch=False, **kargs):
        if self.batch_indices_generator is None:
            self.batch_indices_generator = batch_indices_generator(H * W, self.nerf_bs)
        self.epoch.data += 1
        batch_size = self.nerf_bs
        global_loss = 0
        self.reset()
        if max_f > 1:
            self.initialize_particles(False, rays_o[0,0].cuda())
        else:
            self.initialize_particles(True, rays_o[0,0].cuda())
        self.dynamic_observer.initialize(self.init_particles, self.init_features, self.init_velocities, self.init_rhos, self.init_mu, self.init_lam, self.nerf.voxel_size, self.init_yield_stress, self.init_plastic_viscosity, self.init_friction_alpha, self.cohesion)
        pbar = trange(max_f)
        pbar.set_description(f"[Forward]")
        for i in pbar:
            # save geometry
            if self.write_out and self.epoch.item() % saving_freq == 10:
                pos_and_color = np.concatenate([self.init_particles.data.cpu().numpy(), self.particle_rgb.data.cpu().numpy()], axis=-1)
                os.makedirs(os.path.join(self.base_dir, "geometry"), exist_ok=True)
                write_point_cloud(os.path.join(self.base_dir, f"geometry/{self.epoch.item()}.ply"), pos_and_color)
            feature_grid, particle_pos, xyz_min, xyz_max, min_index, max_index = self.dynamic_observer.forward(i)
            if not self.dynamic_observer.succeed():
                global_loss = torch.tensor(torch.nan, device=rays_o.device)
                break
            observation = dict(grid_observation=feature_grid,
                xyz_min=xyz_min, xyz_max=xyz_max,
                min_index=min_index, max_index=max_index,
                particle_pos=particle_pos)
            self.observations.append(observation)
            if self.write_out:
                self.dynamic_observer.write_out(i, particle_pos.data.cpu(), np.repeat(self.init_alphas.data.cpu()[:, None], 3, axis=1))
            
            if full_batch:
                cam_id_list = list(range(rays_o.shape[0]))
            else:
                cam_id_list = [torch.randint(rays_o.shape[0], [1]).item()]
            
            for cam_id in cam_id_list:
                if partial:
                    indices = next(self.batch_indices_generator)
                    ro = rays_o[cam_id][indices].to(self.init_particles.device)
                    rd = rays_d[cam_id][indices].to(self.init_particles.device)
                    vd = viewdirs[cam_id][indices].to(self.init_particles.device)
                    ta = target[cam_id][i][indices].to(self.init_particles.device)
                    m = rays_mask[cam_id][indices].to(self.init_particles.device)
                    outputs = self.nerf(rays_o=ro, rays_d=rd, viewdirs=vd, 
                                            density_grid=observation['grid_observation'][None, 0:1,...].contiguous(), 
                                            feature_grid=observation['grid_observation'][None, 1:,...].contiguous(), 
                                            **observation, bg=bg, render_depth=False, **kargs)
                    loss = F.mse_loss(m[..., None] * outputs["rgb_marched"], m[..., None] * ta)
                    pout = outputs['alphainv_last'].clamp(1e-5, 1-1e-5)
                    entropy_last_loss = -(pout*torch.log(pout) + (1-pout)*torch.log(1-pout)).mean()
                    loss += self.entropy_weight * entropy_last_loss
                    global_loss += loss.clone().detach()
                    loss.backward()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                else:
                    render_result_chunks = []
                    for ro, rd, vd, ta, m in zip(rays_o[cam_id].split(batch_size, 0), rays_d[cam_id].split(batch_size, 0), viewdirs[cam_id].split(batch_size, 0), target[cam_id][i].split(batch_size, 0), rays_mask[cam_id].split(batch_size, 0)):
                        ro = ro.to(self.init_particles.device)
                        rd = rd.to(self.init_particles.device)
                        vd = vd.to(self.init_particles.device)
                        ta = ta.to(self.init_particles.device)
                        m = m.to(self.init_particles.device)
                        outputs = self.nerf(rays_o=ro, rays_d=rd, viewdirs=vd, 
                                            density_grid=observation['grid_observation'][None, 0:1,...], 
                                            feature_grid=observation['grid_observation'][None, 1:,...], 
                                            **observation, bg=bg, render_depth=True, **kargs)
                        loss = (m[...,None] * (outputs["rgb_marched"] - ta) ** 2).sum() / rays_o.shape[1] / max_f
                        global_loss += loss.clone().detach()
                        loss.backward()
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        # if self.epoch.item() % saving_freq == 0:
                        render_result_chunks.append({k: outputs[k].data.cpu() for k in outputs.keys()})
                
                pbar.set_description(f"[Forward] loss: {global_loss.item()}")

        return global_loss
        
    def backward(self, max_f, **kargs):
        if not self.dynamic_observer.succeed():
            return
        pbar = trange(max_f)
        pbar.set_description(f"[Backward]")
        # max_grad_norm = 0
        # index = -1
        for ri in pbar:
            i = max_f - 1 - ri
            with torch.no_grad():
                obs_grad = self.observations[i]['grid_observation'].grad
                pos_grad = self.observations[i]['particle_pos'].grad
            if i > 0:
                self.dynamic_observer.backward(i, obs_grad, pos_grad, self.observations[i]['min_index'])
            else:
                feature_grad, _, velocity_grad, rho_grad, \
                    mu_grad, lam_grad, yield_stress_grad, viscosity_grad, \
                    friction_alpha_grad, cohesion_grad \
                        = self.dynamic_observer.backward(i, obs_grad, pos_grad, self.observations[i]['min_index'])
                if max_f == 1:
                    volume_loss = self.volume_weight * self.init_alphas.clamp(1e-4, 1e-1).sum() * self.dynamic_observer.dx[None] ** 2 / 8
                    volume_loss.backward(retain_graph=True)
                    self.nerf.density_total_variation_add_grad(self.tv_weight, True)
                self.init_features.backward(retain_graph=True, gradient=feature_grad)
                self.init_velocities.backward(retain_graph=True, gradient=velocity_grad)
                self.init_rhos.backward(retain_graph=True, gradient=rho_grad)
                self.init_mu.backward(retain_graph=True, gradient=mu_grad)
                self.init_lam.backward(retain_graph=True, gradient=lam_grad)
                self.init_yield_stress.backward(retain_graph=True, gradient=yield_stress_grad)
                self.init_plastic_viscosity.backward(retain_graph=True, gradient=viscosity_grad)
                self.init_friction_alpha.backward(retain_graph=True, gradient=friction_alpha_grad)
                self.cohesion.backward(gradient=cohesion_grad)


    @torch.no_grad()
    def render_sequence(self, max_frame, H, W, c2w=None, K=None,
                      savedir=None, dump_images=False,
                      bg=1, rays_o=None, rays_d=None, viewdirs=None, **kargs):
        '''Render images for the given viewpoints; run evaluation if gt given.
        '''

        to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
        device = torch.device("cuda:0")

        if rays_o is None:
            c2w = torch.tensor(c2w)
            K = torch.tensor(K)
            rays_o, rays_d, viewdirs = get_rays_of_a_view(H, W, K, c2w)
            rays_o = rays_o.flatten(0,-2).to(device)
            rays_d = rays_d.flatten(0,-2).to(device)
            viewdirs = viewdirs.flatten(0,-2).to(device)
        else:
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        rgbs = []
        depths = []
        bgmaps = []

        batch_size= self.nerf_bs
        self.reset()
        self.initialize_particles(ray_o=rays_o[0])
       
        self.dynamic_observer.initialize(self.init_particles, self.init_features, self.init_velocities, self.init_rhos, self.init_mu, self.init_lam, self.nerf.voxel_size, self.init_yield_stress, self.init_plastic_viscosity, self.init_friction_alpha, self.cohesion)
        for i in tqdm(range(max_frame)):
            t = i * self.frame_dt
            render_result_chunks = []

            feature_grid, particle_pos, xyz_min, xyz_max, min_index, max_index = self.dynamic_observer.forward(i)
            if self.write_out:
                pos_and_color = np.concatenate([particle_pos.data.cpu().numpy(), np.repeat(self.init_alphas.data.cpu()[:, None], 3, axis=1)], axis=-1)
                os.makedirs(os.path.join(self.base_dir, "simulation"), exist_ok=True)
                mask = self.init_alphas.cpu().numpy() > 0.5
                write_point_cloud(os.path.join(self.base_dir, f"simulation/{i}.ply"), pos_and_color[mask])
            for ro, rd, vd in zip(rays_o.split(batch_size, 0), rays_d.split(batch_size, 0), viewdirs.split(batch_size, 0)):
                outputs = self.nerf(t=t, rays_o=ro, rays_d=rd, viewdirs=vd, 
                                    density_grid=feature_grid[None, 0:1,...].contiguous(), 
                                    feature_grid=feature_grid[None, 1:,...].contiguous(), 
                                    xyz_min=xyz_min, xyz_max=xyz_max, bg=bg, render_depth=True)
                torch.cuda.empty_cache()
                render_result_chunks.append(outputs)
            
            keys = ['rgb_marched', 'depth', 'alphainv_last']
            render_result = {
                k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H,W,-1).cpu().numpy()
                for k in keys
            }
            rgb = render_result['rgb_marched']
            depth = render_result['depth']
            bgmap = render_result['alphainv_last']

            rgbs.append(rgb)
            depths.append(depth)
            bgmaps.append(bgmap)

        if savedir is not None and dump_images:
            os.makedirs(savedir, exist_ok=True)
            for i in trange(len(rgbs)):
                rgb8 = to8b(rgbs[i])
                filename = os.path.join(savedir, '{:03d}.png'.format(i))
                imageio.imwrite(filename, rgb8)

        rgbs = np.array(rgbs)
        depths = np.array(depths)
        bgmaps = np.array(bgmaps)

        return rgbs, depths, bgmaps