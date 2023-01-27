from lib.engine import MPMSimulator
import numpy as np

_base_ = '../configs/default.py'

cfg = dict(
    n_cameras=11,
    n_frames=13,
    dt = 1/4800,
    xyz_min = [-0.4, 0.1, -0.4],
    xyz_max = [0.4, 1.0, 0.4],
    material = MPMSimulator.viscous_fluid,
    pg_scale = [1000, 2000, 4000],
    data_dir = 'data/droplet',
    base_dir = 'checkpoint/droplet',
    dx = 0.16,
    cuda_chunk_size = 10,
    nerf_bs = 2**17,
    particle_chunk_size=2**10,
    kappa = 1e3,
    mu = 1.0,
    rho = 1000,
    entropy_weight = 0.001,
    volume_weight = 0.0001,
    H = 800,
    W = 800,
    N_static = 6001,
    N_dynamic = 201,
    write_out = True,
    physical_params = dict(global_mu=0.1, global_kappa=0.1),
    hit_frame = 4,
    taichi_cuda_memory=0.65,
)

del MPMSimulator
del np