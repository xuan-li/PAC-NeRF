import numpy as np
import torch
from tqdm import tqdm, trange
from .dvgo import get_rays_of_a_view

def camera_intrinsic(f_mm, H, W, ap_mm):
    pixel_size = ap_mm / W
    f = f_mm / pixel_size
    K = np.array([[f, 0, W/2], [0, f, H/2], [0,0,1]])
    return K

rot_phi = lambda phi : torch.Tensor([
    [1,0,0],
    [0,np.cos(phi),-np.sin(phi)],
    [0,np.sin(phi), np.cos(phi)]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th)],
    [0,1,0],
    [np.sin(th),0, np.cos(th)]]).float()

def pose_spherical(point):
    point = np.array(point).astype(np.float32)
    point[1:3] *= -1
    radius = np.linalg.norm(point)
    point /= radius
    x,y,z = point
    xz = np.array([x, 0, z])
    source_vector = np.array([0., 0., -1.])
    if np.linalg.norm(xz) < 1e-6:
        Rx = 0.5 * np.sign(y) * np.pi
        Ry = 0
    else:
        xz /= np.linalg.norm(xz)
        cosRx = np.dot(xz, point)
        axis = np.cross(source_vector, xz)
        cosRy = np.dot(source_vector, xz)     
        Rx = np.arccos(cosRx)
        Ry = np.arccos(cosRy)
        if y < 0:
            Rx = -Rx
        if axis[1] < 0:
            Ry = -Ry
    R = rot_theta(Ry) @ rot_phi(Rx)
    T = -radius * np.dot(R, source_vector[:, None])
    c2w = torch.tensor(np.concatenate([R, T], axis=-1)).float()
    return c2w