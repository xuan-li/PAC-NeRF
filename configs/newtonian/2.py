from lib.engine import MPMSimulator
import numpy as np

_base_ = 'default.py'

cfg = dict(
    data_dir = 'data/newtonian/2',
    base_dir = 'checkpoint/newtonian/2',
)

del MPMSimulator
del np