import taichi as ti
import math
import torch
from ..utils import Timer

@ti.data_oriented
class MPMSimulator:
    # Stick to the boundary
    surface_sticky = 0
    # Slippy boundary
    surface_slip = 1
    # Slippy and free to separate
    surface_separate = 2

    elasticity = 10
    viscous_fluid = 11
    von_mises = 12
    drucker_prager = 13

    def __init__(self, dtype, dt, frame_dt, particle_layout, dx, inv_dx, n_particles, gravity=[0, -9.8, 0], material=elasticity, cuda_chunk_size=400, **kargs):
        
        # Surface boundary conditions
        dim = self.dim = 3
        self.dtype = dtype
        self.material = material
        self.particle = particle_layout
        self.n_particles = n_particles
        self.dx = dx
        self.inv_dx = inv_dx
        self.frame_dt = frame_dt
        self.cfl_satisfy = ti.field(ti.i8, shape=())
        self.p_vol = ti.field(self.dtype, shape=())
        self.dt = ti.field(self.dtype, shape=())
        self.n_substeps = ti.field(ti.i32, shape=())
        self.cuda_chunk_size = cuda_chunk_size
        print("cuda_chunk_size: ", cuda_chunk_size)
        self.step_particle = self.particle.dense(ti.j, cuda_chunk_size+1) # the last one is the first one in the next chunk
        self.x = ti.Vector.field(dim, dtype=self.dtype, needs_grad=True)  # position
        self.v = ti.Vector.field(dim, dtype=self.dtype, needs_grad=True)  # velocity
        self.C = ti.Matrix.field(dim, dim, dtype=self.dtype, needs_grad=True)  # affine velocity field
        self.F = ti.Matrix.field(dim, dim, dtype=self.dtype, needs_grad=True)  # deformation gradient
        self.step_particle.place(self.x,
                            self.x.grad, 
                            self.v,
                            self.v.grad,
                            self.C,
                            self.C.grad,
                            self.F,
                            self.F.grad)
        
        self.damping_coeff = 0

        # material
        self.friction_alpha = ti.field(dtype=self.dtype, shape=(), needs_grad=True)
        self.cohesion = ti.field(dtype=self.dtype, shape=(), needs_grad=True)
        self.yield_stress = ti.field(dtype=self.dtype, shape=(), needs_grad=True)
        self.plastic_viscosity = ti.field(dtype=self.dtype, shape=(), needs_grad=True)
        self.mu = ti.field(dtype=self.dtype, needs_grad=True)
        self.lam = ti.field(dtype=self.dtype, needs_grad=True)
        self.p_mass = ti.field(self.dtype, needs_grad=True)
        self.F_tmp = ti.Matrix.field(dim, dim, dtype=self.dtype, needs_grad=True)  # deformation gradient
        self.U = ti.Matrix.field(dim, dim, dtype=self.dtype, needs_grad=True)
        self.V = ti.Matrix.field(dim, dim, dtype=self.dtype, needs_grad=True)
        self.sig = ti.Matrix.field(dim, dim, dtype=self.dtype, needs_grad=True)
        self.sig_out = ti.Vector.field(dim, dtype=self.dtype, needs_grad=True) # for isotropic matrials

        self.particle.place(self.mu,
                            self.mu.grad, 
                            self.lam,
                            self.lam.grad,
                            self.p_mass,
                            self.p_mass.grad,
                            self.F_tmp,
                            self.F_tmp.grad, 
                            self.U,
                            self.U.grad,
                            self.V,
                            self.V.grad,
                            self.sig,
                            self.sig.grad,
                            self.sig_out,
                            self.sig_out.grad
                            )

        grid_size = 4096
        offset = self.offset = tuple(-grid_size // 2 for _ in range(3))
        self.offset_vec = ti.Vector(list(offset), ti.i32)
        grid_block_size = 128
        leaf_block_size = self.leaf_block_size = 4
        
        grid = self.grid = ti.root.pointer(ti.ijk, grid_size // grid_block_size)
        block = self.block = grid.pointer(ti.ijk, grid_block_size // leaf_block_size)

        self.grid_m = ti.field(dtype=self.dtype, needs_grad=True)  # grid node mass
        self.grid_v_in = ti.Vector.field(dim, dtype=self.dtype, needs_grad=True)  # grid node momentum/velocity
        self.grid_v_out = ti.Vector.field(dim, dtype=self.dtype, needs_grad=True)  # grid node momentum/velocity
       
        def block_component(c):
            block.dense(ti.ijk, leaf_block_size).place(c, c.grad, offset=offset)

        block_component(self.grid_m)
        block_component(self.grid_v_in)
        block_component(self.grid_v_out)
        
        self.gravity = ti.Vector.field(dim, self.dtype, shape=())# gravity ...
        self.gravity[None] = gravity
        self.dt[None] = dt
        self.n_substeps[None] = round(frame_dt / dt)

        self.analytic_collision = []
        self.cached_states = []
    
    def set_dt(self, dt):
        self.dt[None] = dt
        self.n_substeps[None] = round(self.frame_dt / dt)
    
    def clear_grads(self):
        self.mu.grad.fill(0) 
        self.lam.grad.fill(0) 
        self.p_mass.grad.fill(0)
        self.x.grad.fill(0)
        self.v.grad.fill(0)
        self.C.grad.fill(0)
        self.F.grad.fill(0)
        self.yield_stress.grad.fill(0)
        self.plastic_viscosity.grad.fill(0)
        self.friction_alpha.grad.fill(0)
        self.cohesion.grad.fill(0)

    def clear_svd_grads(self):
        self.F_tmp.grad.fill(0)
        self.U.grad.fill(0)
        self.V.grad.fill(0)
        self.sig.grad.fill(0)
        self.sig_out.grad.fill(0)
    
    @ti.func
    def smu(self, x1, x2, mu=1e-4):
        return 0.5 * ((x1 + x2) + ti.sqrt((x1 - x2) ** 2 + mu))
    
    @ti.kernel
    def compute_F_tmp(self, s: ti.i32):
        for p in range(self.n_particles[None]):
            if ti.static(self.material==self.viscous_fluid):
                self.F_tmp[p][0,0] = (1.0 + self.dt[None] * self.C[p, s].trace()) * self.F[p, s][0,0]
            else:
                self.F_tmp[p] = (ti.Matrix.identity(self.dtype, self.dim) + self.dt[None] * self.C[p, s]) @ self.F[p, s]

    @ti.kernel
    def svd(self):
        for p in range(self.n_particles[None]):
            self.U[p], self.sig[p], self.V[p] = ti.svd(self.F_tmp[p].cast(ti.f64))

    @ti.kernel
    def svd_grad(self):
        for p in range(self.n_particles[None]):
            self.F_tmp.grad[p] += self.backward_svd(self.U.grad[p].cast(ti.f64), self.sig.grad[p].cast(ti.f64), self.V.grad[p].cast(ti.f64), self.U[p].cast(ti.f64), self.sig[p].cast(ti.f64), self.V[p].cast(ti.f64))

    @ti.kernel
    def project_F(self, s: ti.i32):
        for p in range(self.n_particles[None]):
            sig = ti.Vector([ti.max(self.sig[p][0,0], 0.05), ti.max(self.sig[p][1,1], 0.05), ti.max(self.sig[p][2,2], 0.05)])
            # sig = ti.Vector([self.sig[p][0,0], self.sig[p][1,1], self.sig[p][2,2]])
            if ti.static(self.material == self.viscous_fluid):
                self.F[p, s+1][0,0] = ti.max(self.F_tmp[p][0,0], 0.05)
                # self.F[p, s+1][0,0] = self.F_tmp[p][0,0]
            elif ti.static(self.material == self.drucker_prager):
                epsilon = ti.log(sig)
                trace_epsilon = epsilon.sum()
                shifted_trace = trace_epsilon - self.cohesion[None] * self.dim
                if shifted_trace >= 0:
                    epsilon = ti.Vector.one(self.dtype, self.dim) * self.cohesion[None]
                else:
                    epsilon_hat = epsilon - (epsilon.sum() / self.dim)
                    epsilon_hat_norm = self.norm(epsilon_hat)
                    delta_gamma = epsilon_hat_norm + (self.dim * self.lam[p] + 2. * self.mu[p]) / (2. * self.mu[p]) * (shifted_trace) * self.friction_alpha[None]
                    epsilon -= (ti.max(delta_gamma, 0) / epsilon_hat_norm) * epsilon_hat
                sig_out = ti.exp(epsilon)
                self.sig_out[p] = sig_out
                self.F[p, s+1] = self.U[p] @ self.make_matrix_from_diag(sig_out) @ self.V[p].transpose()
            elif ti.static(self.material == self.von_mises):
                b_trial = sig ** 2
                epsilon = ti.log(sig)
                trace_epsilon = epsilon.sum()
                epsilon_hat = epsilon - (epsilon.sum() / self.dim)
                s_trial = 2 * self.mu[p] * epsilon_hat
                s_trial_norm = self.norm(s_trial)
                y = s_trial_norm - ti.sqrt(2./3) * self.yield_stress[None]
                sig_out = ti.Vector.zero(self.dtype, self.dim)
                if y > 0:
                    mu_hat = self.mu[p] * b_trial.sum() / self.dim
                    s_new_norm = s_trial_norm - y / (1 + self.plastic_viscosity[None] / (2 * mu_hat * self.dt[None]))
                    s_new = (s_new_norm / s_trial_norm) * s_trial
                    H = s_new / (2 * self.mu[p]) + trace_epsilon / self.dim
                    sig_out = ti.exp(H)
                else:
                    sig_out = sig
                self.sig_out[p] = sig_out
                self.F[p, s+1] = self.U[p] @ self.make_matrix_from_diag(sig_out) @ self.V[p].transpose()
            else:
                self.F[p, s+1] = self.F_tmp[p]


    @ti.func
    def backward_svd(self, gu, gsigma, gv, u, sig, v):
        # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
        vt = v.transpose()
        ut = u.transpose()
        sigma_term = u @ gsigma @ vt
        s = ti.Vector([sig[0, 0], sig[1, 1], sig[2, 2]]) ** 2
        F = ti.Matrix.zero(ti.f64, self.dim, self.dim)
        for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
            if i != j: F[i, j] = 1./self.clamp(s[j] - s[i])
        u_term = u @ ((F * (ut@gu - gu.transpose()@u)) @ sig) @ vt
        v_term = u @ (sig @ ((F * (vt@gv - gv.transpose()@v)) @ vt))
        return u_term + v_term + sigma_term

    @ti.func
    def make_matrix_from_diag(self, d):
        if ti.static(self.dim==2):
            return ti.Matrix([[d[0], 0.0], [0.0, d[1]]], dt=self.dtype)
        else:
            return ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]], dt=self.dtype)

    @ti.func
    def clamp(self, a, eps=1e-6):
        # remember that we don't support if return in taichi
        # stop the gradient ...
        if a>=0:
            a = ti.max(a, eps)
        else:
            a = ti.min(a, -eps)
        return a

    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

    @ti.kernel
    def p2g(self, s: ti.i32):
        ti.block_local(self.grid_m)
        ti.block_local(self.grid_v_in)
        ti.block_local(self.grid_v_out)
        for p in range(self.n_particles[None]):
            base = ti.floor(self.x[p, s] * self.inv_dx[None] - 0.5).cast(int)
            fx = self.x[p, s] * self.inv_dx[None] - base.cast(self.dtype)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_F = self.F[p, s+1]
            stress = ti.Matrix.zero(self.dtype, self.dim, self.dim)
            if ti.static(self.material == self.elasticity):
                J = new_F.determinant()
                scale = self.lam[p] * ti.log(J) - self.mu[p]
                grad_v = self.C[p, s]
                epsilon = 0.5 * (grad_v + grad_v.transpose())
                stress = self.damping_coeff * epsilon * J + self.mu[p] * (new_F @ new_F.transpose()) + scale * ti.Matrix.identity(self.dtype, self.dim)
            elif ti.static(self.material == self.viscous_fluid):
                J = new_F[0,0]
                kappa = .6666666666 * self.mu[p] + self.lam[p]
                stress = kappa * ti.Matrix.identity(self.dtype, self.dim) * (J - 1 / (J ** 6))
                grad_v = self.C[p, s]
                epsilon = 0.5 * (grad_v + grad_v.transpose())
                stress += self.mu[p] * epsilon * J
            else:
                log_sig = ti.log(self.sig_out[p])
                tau = 2 * self.mu[p] * log_sig + self.lam[p] * log_sig.sum()
                stress = self.U[p] @ self.make_matrix_from_diag(tau) @ self.U[p].transpose()

            stress = (-self.dt[None] * self.p_vol[None] * 4 * self.inv_dx[None]**2) * stress
            affine = stress + self.p_mass[p] * self.C[p, s]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(ti.Vector([i, j, k]), self.dtype) - fx) * self.dx[None]
                        weight = w[i](0) * w[j](1) * w[k](2)
                        self.grid_v_in[base + offset] += \
                            weight * (self.p_mass[p] * self.v[p, s] + affine @ dpos)
                        self.grid_m[base + offset] += weight * self.p_mass[p]
        
    @ti.kernel
    def grid_op(self, s: ti.i32):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 1e-8 * self.dx[None] ** 3:  # No need for epsilon here, 1e-10 is to prevent potential numerical problems ..
                v_out = self.grid_v_in[I] / self.grid_m[I] + self.dt[None] * self.gravity[None]
                for i in ti.static(range(len(self.analytic_collision))): # Caution!! len(self.analytic_collision) will become static once executed
                    v_out = self.analytic_collision[i](I, v_out)
                self.grid_v_out[I] = v_out

    @ti.kernel
    def g2p(self, f: ti.i32): # grid to particle (G2P)
        ti.block_local(self.grid_m)
        ti.block_local(self.grid_v_in)
        ti.block_local(self.grid_v_out)
        for p in range(self.n_particles[None]):
            base = ti.floor(self.x[p, f] * self.inv_dx[None] - 0.5).cast(int)
            fx = self.x[p, f] * self.inv_dx[None] - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(self.dtype, self.dim)
            new_C = ti.Matrix.zero(self.dtype, self.dim, self.dim)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        dpos = ti.cast(ti.Vector([i, j, k]), self.dtype) - fx
                        g_v = self.grid_v_out[base(0) + i, base(1) + j, base(2) + k]
                        weight = w[i](0) * w[j](1) * w[k](2)
                        new_v += weight * g_v
                        new_C += 4 * weight * g_v.outer_product(dpos) * self.inv_dx[None]

            self.v[p, f + 1] = new_v
            self.x[p, f + 1] = self.x[p, f] + self.dt[None] * self.v[p, f + 1]
            self.C[p, f + 1] = new_C
    
    @ti.kernel
    def check_cfl(self, s: ti.i32):
        for p in range(self.n_particles[None]):
            if ti.math.isnan(self.v[p, s]).any():
                self.cfl_satisfy[None] = 0
            if ti.abs(self.v[p, s]).max() * self.dt[None] > self.dx[None]:
                self.cfl_satisfy[None] = 0

    def substep(self, s, cache=True):
        local_index = s % self.cuda_chunk_size
        self.grid.deactivate_all()
        self.compute_F_tmp(local_index)
        self.svd()
        self.project_F(local_index)

        self.p2g(local_index)
        self.grid_op(local_index)
        self.g2p(local_index)
        self.check_cfl(local_index + 1)
        
        if (local_index == self.cuda_chunk_size-1) and cache:
            self.push_to_memory()
    
    def substep_grad(self, s):
        local_index = s % self.cuda_chunk_size
        if local_index == self.cuda_chunk_size-1:
            self.pop_from_memory()

        self.grid.deactivate_all()
        self.compute_F_tmp(local_index)
        self.svd()
        self.project_F(local_index)
        self.p2g(local_index)
        self.grid_op(local_index)

        self.clear_svd_grads()
        self.g2p.grad(local_index)
        self.grid_op.grad(local_index)
        self.p2g.grad(local_index)
        self.project_F.grad(local_index)
        self.svd_grad()
        self.compute_F_tmp.grad(local_index)
        
    @ti.kernel
    def get_x(self, f:ti.i32, x: ti.types.ndarray()):
        local_index = (f * self.n_substeps[None]) % self.cuda_chunk_size
        for i in range(self.n_particles[None]):
            for d in ti.static(range(3)):
                x[i, d] = ti.cast(self.x[i, local_index][d], ti.f32)
    
    def advance(self, f):
        for i in range(self.n_substeps[None] * f, self.n_substeps[None] * (f+1)):
            if self.cfl_satisfy[None]:
                self.substep(i)
    
    def advance_grad(self, f):
        for i in reversed(range(self.n_substeps[None] * f, self.n_substeps[None] * (f+1))):
            if self.cfl_satisfy[None]:
                self.substep_grad(i)
    
    def add_cylinder_collider(self,
                             start,
                             end,
                             radius,
                             surface=surface_sticky):
        start = ti.Vector(list(start))
        end = ti.Vector(list(end))
        axis = ti.Vector(list(end - start))
        length = axis.norm()
        axis /= length

        @ti.func
        def get_velocity(I, v):
            offset = I.cast(self.dtype) * self.dx[None] - start
            axis_component = offset.dot(axis)
            normal = offset - axis_component * axis
            dist = self.norm(normal)
            n = normal / dist
            if axis_component >= 0 and axis_component <= length and dist <= radius:
                if ti.static(surface == self.surface_sticky):
                    v = ti.Vector.zero(self.dtype, self.dim)
                else:
                    normal_component = n.dot(v)
                    if ti.static(surface == self.surface_slip):
                        # Project out all normal component
                        v = v - n * normal_component
                    else:
                        # Project out only inward normal component
                        v = v - n * min(normal_component, 0)

            return v

        self.analytic_collision.append(get_velocity)

    def add_surface_collider(self,
                             point,
                             normal,
                             surface=surface_sticky):
        point = list(point)
        # Normalize normal
        normal_scale = 1.0 / math.sqrt(sum(x**2 for x in normal))
        normal = list(normal_scale * x for x in normal)

        @ti.func
        def get_velocity(I, v):
            offset = I.cast(self.dtype) * self.dx[None] - ti.Vector(point)
            n = ti.Vector(normal)
            if offset.dot(n) <= 1e-6:
                if ti.static(surface == self.surface_sticky):
                    v = ti.Vector.zero(self.dtype, self.dim)
                else:
                    normal_component = n.dot(v)

                    if ti.static(surface == self.surface_slip):
                        # Project out all normal component
                        v = v - n * normal_component
                    else:
                        # Project out only inward normal component
                        v = v - n * min(normal_component, 0)

            return v

        self.analytic_collision.append(get_velocity)

    @ti.kernel
    def get_state_chunk(self, x: ti.types.ndarray(),
                            v: ti.types.ndarray(),
                            C: ti.types.ndarray(),
                            F: ti.types.ndarray(),
                       ):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(self.dim)):
                x[p, i] = self.x[p, 0][i]
                v[p, i] = self.v[p, 0][i]
                for j in ti.static(range(self.dim)):
                    C[p, i, j] = self.C[p, 0][i, j]
                    F[p, i, j] = self.F[p, 0][i, j]
        
        for p in range(self.n_particles[None]):
            self.x[p, 0] = self.x[p, self.cuda_chunk_size]
            self.v[p, 0] = self.v[p, self.cuda_chunk_size]
            self.C[p, 0] = self.C[p, self.cuda_chunk_size]
            self.F[p, 0] = self.F[p, self.cuda_chunk_size]

    @ti.kernel
    def set_state_chunk(self, x: ti.types.ndarray(),
                            v: ti.types.ndarray(),
                            C: ti.types.ndarray(),
                            F: ti.types.ndarray(),
                       ):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(self.dim)):
                self.x[p, 0][i] = x[p, i]
                self.v[p, 0][i] = v[p, i]
                for j in ti.static(range(self.dim)):
                    self.C[p, 0][i, j] = C[p, i, j]
                    self.F[p, 0][i, j] = F[p, i, j]
    
    @ti.kernel
    def cache_gradient(self, x_grad: ti.types.ndarray(),
                            v_grad: ti.types.ndarray(),
                            C_grad: ti.types.ndarray(),
                            F_grad: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(self.dim)):
                x_grad[p, i] = self.x.grad[p, 0][i]
                v_grad[p, i] = self.v.grad[p, 0][i]
                for j in ti.static(range(self.dim)):
                    C_grad[p, i, j] = self.C.grad[p, 0][i, j]
                    F_grad[p, i, j] = self.F.grad[p, 0][i, j]
    
    @ti.kernel
    def prepare_gradient(self, x_grad: ti.types.ndarray(),
                            v_grad: ti.types.ndarray(),
                            C_grad: ti.types.ndarray(),
                            F_grad: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for i in ti.static(range(self.dim)):
                self.x.grad[p, self.cuda_chunk_size][i] = x_grad[p, i]
                self.v.grad[p, self.cuda_chunk_size][i] = v_grad[p, i]
                for j in ti.static(range(self.dim)):
                    self.C.grad[p, self.cuda_chunk_size][i, j] = C_grad[p, i, j]
                    self.F.grad[p, self.cuda_chunk_size][i, j] = F_grad[p, i, j]

    def push_to_memory(self):
        if self.dtype == ti.f32:
            dtype=torch.float32
        else:
            dtype=torch.float64
        x = torch.zeros([self.n_particles[None], self.dim], dtype=dtype)
        v = torch.zeros([self.n_particles[None], self.dim], dtype=dtype)
        C = torch.zeros([self.n_particles[None], self.dim, self.dim], dtype=dtype)
        F = torch.zeros([self.n_particles[None], self.dim, self.dim], dtype=dtype)
        self.get_state_chunk(x, v, C, F)
        state = dict(
            x=x, 
            v=v,
            C=C,
            F=F
        )
        self.cached_states.append(state)

    def pop_from_memory(self):
        if self.dtype == ti.f32:
            dtype=torch.float32
        else:
            dtype=torch.float64
        x_grad = torch.zeros([self.n_particles[None], self.dim], dtype=dtype)
        v_grad = torch.zeros([self.n_particles[None], self.dim], dtype=dtype)
        C_grad = torch.zeros([self.n_particles[None], self.dim, self.dim], dtype=dtype)
        F_grad = torch.zeros([self.n_particles[None], self.dim, self.dim], dtype=dtype)
        self.cache_gradient(x_grad, v_grad, C_grad, F_grad)
        self.x.grad.fill(0)
        self.v.grad.fill(0)
        self.C.grad.fill(0)
        self.F.grad.fill(0)
        state = self.cached_states.pop()
        self.set_state_chunk(state['x'], state['v'], state['C'], state['F'])
        for i in range(self.cuda_chunk_size):
            self.substep(i, False)
        self.prepare_gradient(x_grad, v_grad, C_grad, F_grad)