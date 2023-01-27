import torch
import os
import imageio
import numpy as np
from lib import PACNeRF
import mmcv

device = torch.device("cuda:0")
torch.manual_seed(123)
np.random.seed(123)

def load_data(data_folder):
    checkpoint = torch.load(os.path.join(data_folder, "data.pt"))
    rays_o_all = checkpoint['rays_o']
    rays_d_all = checkpoint['rays_d']
    viewdirs_all = checkpoint['viewdirs']
    rgb_all = checkpoint['rgb_all']
    return rays_o_all, rays_d_all, viewdirs_all, rgb_all, 

def create_model(cfg, stage, pnerf=None):
    base_dir=cfg['base_dir']
    checkpoint = torch.load(os.path.join(base_dir, f"model/train_{stage}.pt"))    
    saved_cfg = checkpoint['cfg']
    cfg.update(saved_cfg)
    os.makedirs(os.path.join(base_dir, "image"), exist_ok=True)
    if pnerf == None:
        pnerf = PACNeRF(**cfg)
        pnerf.to(device)
    if checkpoint:
        pnerf.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return pnerf

import argparse
def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    parser.add_argument('--num-frame', required=True, type=int,
                        help='num of frames')
    parser.add_argument('--cam-id', required=True, type=int,
                        help='camera id')
    return parser

if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)['cfg']
    rays_o_all, rays_d_all, viewdirs_all, rgb_all = load_data(cfg['data_dir'])
    pnerf = create_model(cfg, 'dynamic')
    save_dir=os.path.join(cfg['base_dir'],f'image')
    rgbs, depths, bgmaps = pnerf.render_sequence(args.num_frame, H=cfg["H"], W=cfg['W'], c2w=None, K=None, rays_o=rays_o_all[args.cam_id], rays_d=rays_d_all[args.cam_id], viewdirs=viewdirs_all[args.cam_id], savedir=save_dir, dump_images=True)
    to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)
    imageio.mimwrite(os.path.join(cfg['base_dir'],f'video_{args.cam_id}.rgb.mp4'), to8b(rgbs), fps=24, quality=8)