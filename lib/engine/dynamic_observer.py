import torch
import taichi as ti
from .mpm_simulator import MPMSimulator
import numpy as np
from .mesh_io import write_point_cloud
import os

@ti.data_oriented
class DynamicObserver:
    param_map = dict(
        yield_stress=1,
        friction_alpha=2,
    )
    def __init__(self, dtype, rgbnet_dim, dt, frame_dt, particle_chunk_size=2**14, cuda_chunk_size=400, gravity=[0,0,0], material=MPMSimulator.elasticity, base_dir=None, tensor_float64=False, **kargs):
        self.base_dir = base_dir
        if base_dir:
            os.makedirs(os.path.join(self.base_dir, "simulation"), exist_ok=True)
        self.dtype = ti.f64 if dtype == 'float64' else ti.f32
        self.dx = ti.field(self.dtype, shape=())
        self.inv_dx = ti.field(self.dtype, shape=())
        self.feature_dim = 1 + rgbnet_dim # the first bit is rendering density
        self.num_particles = ti.field(ti.i32, shape=())
        self.particle_feature = ti.field(dtype=self.dtype, needs_grad=True)
        self.particle_rho = ti.field(dtype=self.dtype, needs_grad=True)
        self.particle = ti.root.dynamic(ti.i, 2**30, particle_chunk_size)
        self.particle.place(self.particle_rho, self.particle_rho.grad)
        particle_feature_channel = self.particle.dense(ti.j, self.feature_dim)
        particle_feature_channel.place(self.particle_feature, self.particle_feature.grad)

        #intermediate fields
        self.min_index_obs = ti.Vector.field(3, ti.i32, shape=())
        self.max_index_obs = ti.Vector.field(3, ti.i32, shape=())

        # output fields
        self.feature_in = ti.field(dtype=self.dtype, needs_grad=True)
        self.feature_observe = ti.field(dtype=self.dtype, needs_grad=True)
        self.weight_sum = ti.field(dtype=self.dtype, needs_grad=True)

        grid_size = 4096
        offset = tuple(-grid_size // 2 for _ in range(3))
        grid_block_size = 128
        leaf_block_size = 4

        grid_observe = self.grid_observe = ti.root.pointer(ti.ijk, grid_size // grid_block_size)
        block_observe = grid_observe.pointer(ti.ijk, grid_block_size // leaf_block_size)

        def block_observe_component(c):
            block_observe.dense(ti.ijk, leaf_block_size).place(c, c.grad, offset=offset)
        
        def multi_chennel_block_observe_component(c):
            block_observe.dense(ti.ijkl, (leaf_block_size, leaf_block_size, leaf_block_size, self.feature_dim)).place(c, c.grad, offset=offset + (0,))

        block_observe_component(self.weight_sum)
        multi_chennel_block_observe_component(self.feature_in)
        multi_chennel_block_observe_component(self.feature_observe)

        self.simulator = MPMSimulator(dtype=self.dtype, dt=dt, frame_dt=frame_dt, n_particles=self.num_particles,
                                      material=material, dx=self.dx, inv_dx=self.inv_dx, 
                                      particle_layout=self.particle, gravity=gravity, 
                                      cuda_chunk_size=cuda_chunk_size)
        self.device = None
        self.tensor_float64 = tensor_float64
        self.weight_sum_thres = 1e-10


    def get_kwargs(self):
        return {}

    def succeed(self):
        return self.simulator.cfl_satisfy[None]
    
    @ti.kernel
    def compute_particle_mass(self):
        for p in range(self.num_particles[None]):
            self.simulator.p_mass[p] = self.particle_rho[p] * self.simulator.p_vol[None]
    
    @ti.kernel
    def from_torch(self, particles: ti.types.ndarray(), 
                         features: ti.types.ndarray(), 
                         velocities: ti.types.ndarray(), 
                         particle_rho: ti.types.ndarray(), 
                         particle_mu: ti.types.ndarray(), 
                         particle_lam: ti.types.ndarray()):
        # assume cell is indexed by the bottom corner
        for p in range(self.num_particles[None]):
            self.particle_rho[p] = particle_rho[p]
            self.simulator.mu[p] = particle_mu[p]
            self.simulator.lam[p] = particle_lam[p]
            self.simulator.p_mass[p] = 0
            self.simulator.F[p, 0] = ti.Matrix.identity(self.dtype, 3)
            self.simulator.C[p, 0] = ti.Matrix.zero(self.dtype, 3, 3)
            for d in ti.static(range(3)):
                self.simulator.x[p, 0][d] = particles[p, d]
                self.simulator.v[p, 0][d] = velocities[p, d]
            for d in range(self.feature_dim):
                self.particle_feature[p, d] = features[p, d]
    
    @ti.kernel
    def _get_obs(self, f:ti.i32, feature_grid:ti.types.ndarray()):
        for I in ti.grouped(self.weight_sum):
            if self.weight_sum[I] > self.weight_sum_thres:
                i, j, k = I # taichi index
                it, jt, kt = I - self.min_index_obs[None] # torch index
                for d in range(self.feature_dim):
                    feature_grid[d, it, jt, kt] = self.feature_observe[i, j, k, d]

    @ti.kernel
    def set_obs_grad(self, f: ti.i32, dLdc: ti.types.ndarray()):
        for I in ti.grouped(self.weight_sum):
            i, j, k = I
            ir, jr, kr = I - self.min_index_obs[None] # torch index
            if self.weight_sum[I] > self.weight_sum_thres:
                for d in range(self.feature_dim):
                    self.feature_observe.grad[i, j, k, d] += dLdc[d, ir, jr, kr]
    
    @ti.kernel
    def get_bbox(self):
        self.min_index_obs[None] = [4096, 4096, 4096]
        self.max_index_obs[None] = [-4096, -4096, -4096]
        for I in ti.grouped(self.weight_sum):
            if self.weight_sum[I] > self.weight_sum_thres:
                for d in ti.static(range(3)):
                    ti.atomic_min(self.min_index_obs[None][d], I[d])
                    ti.atomic_max(self.max_index_obs[None][d], I[d])

    @ti.kernel
    def get_input_grad(self, feature_grad: ti.types.ndarray(), position_grad: ti.types.ndarray(), 
                             velocity_grad: ti.types.ndarray(), rho_grad: ti.types.ndarray(), 
                             mu_grad: ti.types.ndarray(), lam_grad: ti.types.ndarray()):
        for p in range(self.num_particles[None]):
            rho_grad[p] = self.particle_rho.grad[p]
            mu_grad[p] = self.simulator.mu.grad[p]
            lam_grad[p] = self.simulator.lam.grad[p]
            for d in range(self.feature_dim):
                feature_grad[p, d] = self.particle_feature.grad[p, d]
            for d in ti.static(range(3)):
                velocity_grad[p, d] = self.simulator.v.grad[p, 0][d]
                position_grad[p, d] = self.simulator.x.grad[p, 0][d]
    
    def clear_grads(self):
        self.particle_feature.grad.fill(0)
        self.particle_rho.grad.fill(0)
        self.simulator.clear_grads()
    
    def initialize(self, particles, features, 
                         velocities, particle_rho, 
                         particle_mu, particle_lam, 
                         dx,
                         yield_stress=torch.tensor([0.]), eta=torch.tensor([0.]), 
                         friction_alpha=torch.tensor([0.]), cohesion=torch.tensor([0.])
                         ):
        torch.cuda.synchronize()
        ti.sync()
        self.device = particles.device
        self.num_particles[None] = particles.shape[0]
        self.dx[None], self.inv_dx[None] = dx, 1.0 / dx
        self.simulator.p_vol[None] = (self.dx[None] * 0.5) ** 3
        self.clear_grads()
        self.simulator.cached_states.clear()
        self.from_torch(particles.data.cpu().numpy(), features.data.cpu().numpy(), velocities.data.cpu().numpy(), particle_rho.data.cpu().numpy(), particle_mu.data.cpu().numpy(), particle_lam.data.cpu().numpy())
        self.compute_particle_mass()
        self.simulator.yield_stress[None] = yield_stress.item()
        self.simulator.plastic_viscosity[None] = eta.item()
        self.simulator.friction_alpha[None] = friction_alpha.item()
        self.simulator.cohesion[None] = cohesion.item()
        self.simulator.cfl_satisfy[None] = True
    
    def forward(self, f):
        if f > 0:
            self.simulator.advance(f-1)
            if not self.simulator.cfl_satisfy[None]:
                self.simulator.cached_states.clear()
                return None, None, None, None, None, None
        self.grid_observe.deactivate_all()
        self.p2g(f)
        if self.tensor_float64:
            dtype=np.float64
        else:
            dtype=np.float32
        particle_pos = np.zeros([self.num_particles[None], 3], dtype=dtype)
        self.simulator.get_x(f, particle_pos)
        self.get_bbox()
        min_index =  np.array([self.min_index_obs[None][d] for d in range(3)])
        max_index = np.array([self.max_index_obs[None][d] for d in range(3)])
        shape = max_index - min_index + 1
        feature_grid = -100 * np.ones(shape=[self.feature_dim, *shape], dtype=dtype)
        xyz_min_obs =  torch.from_numpy(self.dx[None] * min_index.astype(dtype)).to(self.device)
        xyz_max_obs = torch.from_numpy(self.dx[None] * max_index.astype(dtype)).to(self.device)
        self._get_obs(f, feature_grid)
        feature_grid = torch.from_numpy(feature_grid).to(self.device).requires_grad_()
        particle_pos =torch.from_numpy(particle_pos).to(self.device).requires_grad_()
        return feature_grid, particle_pos, xyz_min_obs, xyz_max_obs, min_index, max_index
    
    def backward(self, f, dLdc, dLdp, min_index):
        self.min_index_obs[None] = min_index
        self.grid_observe.deactivate_all()
        if dLdp is not None:
            self.set_pos_grad(f, dLdp.data.cpu().numpy())
        if dLdc is not None:
            self.p2g_grad(f, dLdc.data.cpu().numpy())
        if f > 0:
            self.simulator.advance_grad(f-1)
        else:
            self.compute_particle_mass.grad()
            if self.tensor_float64:
                dtype=np.float64
            else:
                dtype=np.float32
            feature_grad = np.zeros([self.num_particles[None], self.feature_dim], dtype=dtype)
            velocity_grad = np.zeros([self.num_particles[None], 3], dtype=dtype)
            position_grad = np.zeros([self.num_particles[None], 3], dtype=dtype)
            rho_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            mu_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            lam_grad = np.zeros([self.num_particles[None]], dtype=dtype)
            yield_stress_grad = np.zeros([1], dtype=dtype)
            viscosity_grad = np.zeros([1], dtype=dtype)
            friction_alpha_grad = np.zeros([1], dtype=dtype)
            cohesion_grad = np.zeros([1], dtype=dtype)
            self.get_input_grad(feature_grad, position_grad, velocity_grad, rho_grad, mu_grad, lam_grad)
            yield_stress_grad[0] = self.simulator.yield_stress.grad[None]
            friction_alpha_grad[0] = self.simulator.friction_alpha.grad[None]
            viscosity_grad[0] = self.simulator.plastic_viscosity.grad[None]
            cohesion_grad[0] = self.simulator.cohesion.grad[None]
            return torch.from_numpy(feature_grad).to(self.device), \
                   torch.from_numpy(position_grad).to(self.device), \
                   torch.from_numpy(velocity_grad).to(self.device), \
                   torch.from_numpy(rho_grad).to(self.device), \
                   torch.from_numpy(mu_grad).to(self.device), torch.from_numpy(lam_grad).to(self.device), \
                   torch.from_numpy(yield_stress_grad).to(self.device), torch.from_numpy(viscosity_grad).to(self.device), \
                   torch.from_numpy(friction_alpha_grad).to(self.device), torch.from_numpy(cohesion_grad).to(self.device)
        
    @ti.kernel
    def p2g_weight(self, s: ti.i32):
        for p in range(self.num_particles[None]):
            base = ti.floor(self.simulator.x[p, s] * self.inv_dx[None]).cast(int)
            fx = self.simulator.x[p, s] * self.inv_dx[None] - base.cast(self.dtype)
            # w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            w = [1 - fx, fx]
            for offset in ti.static(ti.ndrange(*((2, ) * 3))):
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(3)):
                    weight *= w[offset[d]][d]
                self.weight_sum[base + offset] += weight
    
    @ti.kernel
    def p2g_feature(self, s: ti.i32, c: ti.i32):
        for p in range(self.num_particles[None]):
            base = ti.floor(self.simulator.x[p, s] * self.inv_dx[None]).cast(int)
            fx = self.simulator.x[p, s] * self.inv_dx[None] - base.cast(self.dtype)
            # w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            w = [1 - fx, fx]
            for offset in ti.static(ti.ndrange(*((2, ) * 3))):
                weight = ti.cast(1.0, self.dtype)
                for d in ti.static(range(3)):
                    weight *= w[offset[d]][d]
                i, j, k = base + offset
                self.feature_in[i, j, k, c] += weight * self.particle_feature[p, c]

    @ti.kernel
    def normalization(self):
        for I in ti.grouped(self.weight_sum):
            if self.weight_sum[I] > self.weight_sum_thres:
                i, j, k = I
                for d in range(self.feature_dim):
                    self.feature_observe[i, j, k, d] = self.feature_in[i, j, k, d] / self.weight_sum[i, j, k]
    
    @ti.kernel
    def normalization_grad(self):
        for I in ti.grouped(self.weight_sum):
            if self.weight_sum[I] > self.weight_sum_thres:
                i, j, k = I
                weight_inv = 1.0 / self.weight_sum[I]
                weight_square_inv = weight_inv ** 2
                for d in range(self.feature_dim):
                    self.feature_in.grad[i, j, k, d] += self.feature_observe.grad[i, j, k, d] * weight_inv
                    self.weight_sum.grad[I] += - self.feature_observe.grad[i, j, k, d] * self.feature_in[i, j, k, d] * weight_square_inv

    @ti.kernel
    def set_pos_grad(self, f:ti.i32, dLdpo: ti.types.ndarray()):
        s = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size
        for p in range(self.num_particles[None]):
            for d in ti.static(range(3)):
                self.simulator.x.grad[p, s][d] += dLdpo[p, d]

    def p2g(self, f):
        s = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size
        self.p2g_weight(s)
        for d in range(self.feature_dim):
            self.p2g_feature(s, d)
        self.normalization()

    def p2g_grad(self, f, dLdc):
        s = (f * self.simulator.n_substeps[None]) % self.simulator.cuda_chunk_size
        self.p2g_weight(s)
        for d in range(self.feature_dim):
            self.p2g_feature(s, d)
        self.normalization()

        if dLdc is not None:
            self.set_obs_grad(s, dLdc)
        self.normalization_grad()
        for d in range(self.feature_dim):
            self.p2g_feature.grad(s, d)
        self.p2g_weight.grad(s)

    def write_out(self, f, x, color, prefix=""):
        if self.base_dir:
            pos_and_color = np.concatenate([x, color], axis=-1)
            write_point_cloud(os.path.join(self.base_dir, f"simulation/{prefix}{f}.ply"), pos_and_color)