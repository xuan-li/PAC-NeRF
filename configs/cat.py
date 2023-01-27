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
    data_dir = 'data/cat',
    base_dir = 'checkpoint/cat',
    cuda_chunk_size = 20,
    nerf_bs = 2**17,
    particle_chunk_size=2**10,
    taichi_cuda_memory=0.6,
    dt = 1/4800,
    H = 800,
    W = 800,
    N_static = 6001,
    N_dynamic = 500,
    E = 1e3,
    nu = 0.2,
    yield_stress = 100.,
    entropy_weight = 0.0001,
    # volume_weight = 0.001,
    write_out = True,
    physical_params = dict(global_E=0.1, yield_stress=0.1, global_nu=1e-2),
    hit_frame = 6,
)

del MPMSimulator