from lib.engine import MPMSimulator

_base_ = '../configs/default.py'

cfg = dict(
    dtype='float32',
    n_cameras = 11,
    n_frames = 14,
    xyz_min = [-0.5, 0.0, -0.7],
    xyz_max = [0.5, 1.2, 0.7],
    dx = 0.12,
    material = MPMSimulator.drucker_prager,
    pg_scale = [1000, 2000, 4000],
    data_dir = 'data/trophy',
    base_dir = 'checkpoint/trophy',
    cuda_chunk_size = 100,
    nerf_bs = 2**16,
    particle_chunk_size=2**10,
    taichi_cuda_memory=0.6,
    dt = 1/4800,
    H = 800,
    W = 800,
    N_static = 6001,
    N_dynamic = 200,
    E = 1e6,
    nu = 0.3,
    friction_angle = 10,
    physical_params = dict(friction_angle=1.0),
    entropy_weight = 0.0001,
    volume_weight = 0.001,
    write_out = True,
    hit_frame = 5
)

del MPMSimulator