import numpy as np

def write_point_cloud(fn, pos_and_color):
    num_particles = len(pos_and_color)
    with open(fn, 'wb') as f:
        header = f"""ply
format binary_little_endian 1.0
comment Created by taichi
element vertex {num_particles}
property float x
property float y
property float z
property float red
property float green
property float blue
end_header
"""
        f.write(str.encode(header))
        f.write(pos_and_color.tobytes())
