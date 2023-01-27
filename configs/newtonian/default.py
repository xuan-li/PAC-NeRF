from lib.engine import MPMSimulator
import numpy as np

_base_ = '../default.py'

cfg = dict(
    dtype='float32',
    n_cameras=11,
    n_frames=16,
    dt = 1/4800,
    xyz_min = [-0.5, 0.1, -0.5],
    xyz_max = [0.5, 1.0, 0.5],
    material = MPMSimulator.viscous_fluid,
    pg_scale = [1000, 2000, 4000],
    dx = 0.16,
    cuda_chunk_size = 50,
    nerf_bs = 2**20,
    particle_chunk_size=2**10,
    kappa = 1e4,
    mu = 10,
    rho = 1000,
    N_static = 6001,
    entropy_weight = 1e-4,
    volume_weight = 0.001,
    H = 400,
    W = 400,
    write_out = True,
    physical_params = dict(global_mu=0.1, global_kappa=0.1),
    hit_frame = 5,
    taichi_cuda_memory=0.5,
)

del MPMSimulator
del np