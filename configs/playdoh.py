from lib.engine import MPMSimulator

_base_ = '../configs/default.py'

cfg = dict(
    dtype='float32',
    n_cameras = 11,
    n_frames = 12,
    xyz_min = [-0.5, 0.0, -0.7],
    xyz_max = [0.5, 1.2, 0.7],
    dx = 0.16,
    material = MPMSimulator.von_mises,
    pg_scale = [1000, 2000, 4000],
    data_dir = 'data/playdoh',
    base_dir = 'checkpoint/playdoh',
    cuda_chunk_size = 100,
    nerf_bs = 2**16,
    particle_chunk_size=2**10,
    taichi_cuda_memory=0.6,
    dt = 1/4800,
    H = 800,
    W = 800,
    N_static = 6001,
    N_dynamic = 500,
    E = 1e5,
    nu = 0.4,
    yield_stress = 1e3,
    entropy_weight = 1e-4,
    volume_weight = 1e-3,
    write_out = True,
    physical_params = dict(global_E=1e-1, yield_stress=0.1, global_nu=1e-2),
    hit_frame = 6,
    warmup_step = 2,
)

del MPMSimulator