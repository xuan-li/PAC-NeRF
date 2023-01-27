from lib.engine import MPMSimulator

_base_ = '../default.py'

cfg = dict(
    dtype='float32',
    n_cameras = 11,
    n_frames = 16,
    xyz_min = [-0.5, 0.1, -0.5],
    xyz_max = [0.5, 1.0, 0.5],
    dx = 0.16,
    material = MPMSimulator.von_mises,
    pg_scale = [1000, 2000, 4000],
    cuda_chunk_size = 100,
    nerf_bs = 2**20,
    particle_chunk_size=2**10,
    taichi_cuda_memory=0.6,
    dt = 1/4800,
    H = 400,
    W = 400,
    N_static = 6001,
    kappa = 1e5,
    mu = 100,
    yield_stress = 10,
    plastic_viscosity = 1,
    rho = 1000,
    entropy_weight = 0.0001,
    volume_weight = 0.001,
    write_out = True,
    physical_params = dict(global_mu=0.1, global_kappa=0.1, yield_stress=0.1, plastic_viscosity=0.05),
    hit_frame = 5,
)

del MPMSimulator