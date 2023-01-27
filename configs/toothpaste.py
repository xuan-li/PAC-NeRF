from lib.engine import MPMSimulator
import numpy as np

_base_ = '../configs/default.py'

cfg = dict(
    n_cameras=11,
    n_frames=14,
    frame_dt=1/50,
    dt = 1/50/500,
    xyz_min = [-0.1, 0.0, -0.1],
    xyz_max = [0.1, 0.3, 0.1],
    material = MPMSimulator.von_mises,
    pg_scale = [1000, 2000, 4000],
    dx = 0.016,
    data_dir = 'data/toothpaste',
    base_dir = 'checkpoint/toothpaste',
    cuda_chunk_size = 100,
    nerf_bs = 2**16,
    particle_chunk_size=2**10,
    kappa = 1e4,
    mu = 100,
    yield_stress = 10,
    plastic_viscosity = 0.1,
    rho = 1000,
    entropy_weight = 1e-3,
    volume_weight = 1e-1,
    H = 800,
    W = 800,
    N_static = 6001,
    write_out = True,
    physical_params = dict(global_mu=0.2, global_kappa=0.2, yield_stress=0.2, plastic_viscosity=0.1),
    hit_frame = 6,
    taichi_cuda_memory=0.6,
)

del MPMSimulator
del np