from lib.engine import MPMSimulator

_base_ = 'default.py'

cfg = dict(
    data_dir = 'data/non_newtonian/8',
    base_dir = 'checkpoint/non_newtonian/8',
)

del MPMSimulator