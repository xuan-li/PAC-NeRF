from lib.engine import MPMSimulator

_base_ = '../configs/default.py'

cfg = dict(
    dtype='float32',
    n_cameras = 11,
    n_frames = 14,
    xyz_min = [-0.6, 0.2, -0.4],
    xyz_max = [0.6, 1.0, 0.4],
    dx = 0.02 * 8,
    material = MPMSimulator.elasticity,
    pg_scale = [1000, 2000, 4000],
    data_dir = 'data/bird',
    base_dir = 'checkpoint/bird',
    cuda_chunk_size = 100,
    nerf_bs = 2**17,
    particle_chunk_size=2**10,
    taichi_cuda_memory=0.6,
    dt = 1/24/100,
    H = 800,
    W = 800,
    N_static = 6001,
    N_dynamic = 201,
    entropy_weight = 0.0001,
    volume_weight = 0.001,
    tv_weight = 0,
    direct_nerf = False,
    E = 1e3,
    nu = 0.1,
    rho = 1000,
    hit_frame = 5,
    physical_params = dict(global_E=1e-1, global_nu=1e-2),
    write_out = True,
    random_particles=True,
)

del MPMSimulator