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
    pg_scale = [1000, 3000, 5000],
    cuda_chunk_size = 100,
    nerf_bs = 2**20,
    particle_chunk_size=2**10,
    taichi_cuda_memory=0.6,
    dt = 1/4800,
    H = 400,
    W = 400,
    N_static = 10001,
    E = 1e4,
    nu = 0.25,
    yield_stress = 1000.,
    entropy_weight = 1e-4,
    volume_weight = 1e-4,
    write_out = True,
    physical_params = dict(global_E=1e-1, yield_stress=0.1, global_nu=1e-2),
    hit_frame = 5,
)

del MPMSimulator