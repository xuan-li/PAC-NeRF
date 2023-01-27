from lib.engine import MPMSimulator

_base_ = 'default.py'

cfg = dict(
    data_dir = 'data/sand_batch/2',
    base_dir = 'checkpoint/sand_batch/2',
)

del MPMSimulator