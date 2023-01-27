from lib.engine import MPMSimulator
import numpy as np

_base_ = 'default.py'

cfg = dict(
    data_dir = 'data/newtonian/1',
    base_dir = 'checkpoint/newtonian/1'
)

del MPMSimulator
del np