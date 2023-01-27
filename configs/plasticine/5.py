from lib.engine import MPMSimulator

_base_ = 'default.py'

cfg = dict(
    data_dir = 'data/plasticine_batch/5',
    base_dir = 'checkpoint/plasticine_batch/5',
)

del MPMSimulator