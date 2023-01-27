from lib.engine import MPMSimulator

_base_ = '../configs/default.py'

cfg = dict(
    dtype='float32',
    n_cameras = 11,
    n_frames = 18,
    xyz_min = [-0.4, 0.5, -1.2],
    xyz_max = [0.4, 1.2, 1.2],
    dx = 0.16,
    material = MPMSimulator.elasticity,
    pg_scale = [1000, 2000, 4000],
    data_dir = 'data/rope',
    base_dir = 'checkpoint/rope',
    cuda_chunk_size = 100,
    nerf_bs = 2**17,
    particle_chunk_size=2**10,
    taichi_cuda_memory=0.6,
    dt = 1/24/100,
    H = 800,
    W = 800,
    N_static = 6001,
    entropy_weight = 1e-3,
    volume_weight = 2e-3,
    # tv_weight = 1e-4,
    direct_nerf = False,
    E = 1e3,
    nu = 0.4,
    rho = 1000,
    hit_frame = 5,
    physical_params = dict(global_E=1e-1, global_nu=1e-2),
    write_out = True,
    random_particles=True,
    BC = {"ground": [[0,0,0], [0, 1, 0]], 
          "cylinder0": [[-1.65, 0.5, 0.5], [1.65, 0.5, 0.5], 0.1],
          "cylinder1": [[-1.25, 0.3, -0.6], [1.25, 0.3, -0.6], 0.1]}
)

del MPMSimulator