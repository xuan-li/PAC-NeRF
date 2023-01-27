from lib.engine import MPMSimulator
import numpy as np

_base_ = '../configs/default.py'

cfg = dict(
    n_cameras=11,
    n_frames=14,
    dt = 1/4800,
    xyz_min = [-0.6, 0.2, -0.4],
    xyz_max = [0.6, 1.2, 0.4],
    material = MPMSimulator.von_mises,
    pg_scale = [1000, 2000, 4000],
    dx = 0.12,
    data_dir = 'data/cream',
    base_dir = 'checkpoint/cream',
    cuda_chunk_size = 10,
    nerf_bs = 2**16,
    particle_chunk_size=2**10,
    kappa = 1e4,
    mu = 100,
    yield_stress = 10,
    plastic_viscosity = 1,
    rho = 1000,
    entropy_weight = 1e-4,
    volume_weight = 1e-3,
    H = 800,
    W = 800,
    N_static = 6001,
    write_out = True,
    physical_params = dict(yield_stress=0.1, plastic_viscosity=0.05, global_mu=0.1, global_kappa=0.1),
    hit_frame = 6,
    warm_up_frame = 9,
    taichi_cuda_memory=0.6,
)

del MPMSimulator
del np