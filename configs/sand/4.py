from lib.engine import MPMSimulator

_base_ = 'default.py'

cfg = dict(
    data_dir = 'data/sand_batch/4',
    base_dir = 'checkpoint/sand_batch/4',
)

del MPMSimulator