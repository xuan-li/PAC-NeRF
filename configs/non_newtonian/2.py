from lib.engine import MPMSimulator

_base_ = 'default.py'

cfg = dict(
    data_dir = 'data/non_newtonian/2',
    base_dir = 'checkpoint/non_newtonian/2',
)

del MPMSimulator