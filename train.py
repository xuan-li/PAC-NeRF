import torch
import os
import json
import imageio
import numpy as np
from lib.nerf import get_rays_of_a_view
from lib import PACNeRF, FullBatchLBFGS
import mmcv
import cv2
from tqdm import tqdm
from matting import MattingRefine
import time


time0 = time.time()

device = torch.device("cuda:0")
torch.manual_seed(123)
np.random.seed(123)

def load_data(data_folder, num_views, H, W):
    try:
        checkpoint = torch.load(os.path.join(data_folder, "data.pt"))
        rays_o_all = checkpoint['rays_o']
        rays_d_all = checkpoint['rays_d']
        viewdirs_all = checkpoint['viewdirs']
        rgb_all = checkpoint['rgb_all']
        if 'ray_mask_all' in checkpoint:
            ray_mask_all = checkpoint['ray_mask_all']
        else:
            ray_mask_all = torch.ones([rgb_all.shape[0], rgb_all.shape[2]])
        
    except:
        matting_model = MattingRefine(backbone='resnet101',
                      backbone_scale=1/2,
                      refine_mode = 'sampling',
                      refine_sample_pixels = 100_000)
        matting_model.load_state_dict(torch.load('checkpoint/pytorch_resnet101.pth', map_location=device))
        matting_model = matting_model.eval().to(torch.float32).to(device)
        with open(os.path.join(data_folder, "all_data.json")) as f:
            data_info = json.load(f)

        n_frames = int(len(data_info) / num_views) - 1

        c2w_all = torch.zeros(num_views, 3, 4)
        K_all = torch.zeros(num_views, 3, 3)
        rgb_all = torch.zeros(num_views, n_frames, H, W, 3)
        rays_o_all, rays_d_all, viewdirs_all = torch.zeros(num_views, H, W, 3), torch.zeros(num_views, H, W, 3), torch.zeros(num_views, H, W, 3)
        ray_mask_all = torch.ones(num_views, H, W)
        backgroud_all = np.zeros([num_views, H, W, 3])
        for entry in data_info:
            cam_id, frame_id = [int(i) for i in entry["file_path"].split("/")[-1].rstrip(".png").lstrip("r_").split("_")]
            if frame_id == -1:
                backgroud_all[cam_id] = np.array(imageio.imread(os.path.join(data_folder, entry["file_path"])))[..., :3]
            if frame_id == -2:
                alpha = imageio.imread(os.path.join(data_folder, entry["file_path"]))[..., 3]
                ray_mask_all[cam_id][alpha > 50] = 0
        print("Removing image backgrounds.....")
        for entry in tqdm(data_info):
            cam_id, frame_id = [int(i) for i in entry["file_path"].split("/")[-1].rstrip(".png").lstrip("r_").split("_")]
            if frame_id < 0:
                continue
            c2w_all[cam_id] = torch.tensor(entry["c2w"])
            K_all[cam_id] = torch.tensor(entry["intrinsic"])
            img = np.array(imageio.imread(os.path.join(data_folder, entry["file_path"])))[..., :3]
            bgr = backgroud_all[cam_id]
            with torch.no_grad():
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255).to(device)[None]
                bgr_tensor = torch.from_numpy(bgr.transpose(2, 0, 1).astype(np.float32) / 255).to(device)[None]
                pha = matting_model(img_tensor, bgr_tensor)[0][0,0].cpu().numpy()
                mask = pha < 0.9
            img[mask] = 255
            cv2.imwrite(os.path.join(data_folder, f"data/m_{cam_id}_{frame_id}.png"), img[...,::-1])
            rgb_all[cam_id, frame_id] = torch.from_numpy(img.astype(np.float32) / 255.)

        print("Generating rays.....")
        for i in tqdm(range(num_views)):
            rays_o, rays_d, viewdirs = get_rays_of_a_view(H, W, K_all[i], c2w_all[i])
            rays_o_all[i] = rays_o
            rays_d_all[i] = rays_d
            viewdirs_all[i] = viewdirs

        rays_o_all = rays_o_all.reshape([num_views, -1, 3])
        rays_d_all = rays_d_all.reshape([num_views, -1, 3])
        viewdirs_all = viewdirs_all.reshape([num_views, -1, 3])
        rgb_all = rgb_all.reshape([num_views, n_frames, -1, 3])
        ray_mask_all = ray_mask_all.reshape([num_views, -1])

        print("Caching dataset.....")
        torch.save({
            'rays_o': rays_o_all,
            'rays_d': rays_d_all,
            'viewdirs': viewdirs_all,
            'rgb_all': rgb_all,
            'ray_mask_all': ray_mask_all
        }, os.path.join(data_folder, "data.pt"))

    return rays_o_all, rays_d_all, viewdirs_all, rgb_all, ray_mask_all

def reset_optimizer(stage, pnerf):
    if stage == "static":
        optimizer = torch.optim.Adam(
            [
                {'params': pnerf.nerf.density.parameters(), 'lr': 1e-1},
                {'params': pnerf.nerf.k0.parameters(), 'lr': 1e-1},
                {'params': pnerf.nerf.rgbnet.parameters(), 'lr': 1e-3}],
        )
    elif stage == 'velocity':
        optimizer = FullBatchLBFGS(
            params=[pnerf.global_v],
            lr=1,
            history_size=20,
            debug=True,
            line_search='Wolfe'
        )

    elif stage == 'dynamic':
        optimizer = torch.optim.Adam(
        [ 
            *[{'params': getattr(pnerf, k), 'lr': cfg['physical_params'][k]}
            for k in cfg['physical_params']],
        ],
        amsgrad=False
    )
  
    else:
        optimizer=None
    return optimizer

def create_model_and_optimizer(cfg, stage, pnerf=None):
    base_dir=cfg['base_dir']
    try:
        checkpoint = torch.load(os.path.join(base_dir, f"model/train_{stage}.pt"))    
        saved_cfg = checkpoint['cfg']
        cfg.update(saved_cfg)
        start = checkpoint['epoch']
        print(f"======= {stage} Train from epoch {start} =======")
    except:
        start = 0
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "image"), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "model"), exist_ok=True)
        checkpoint = None
        print(f"======= {stage} Train from Scratch =======")
    if pnerf == None:
        pnerf = PACNeRF(**cfg)
        pnerf.to(device)
    if checkpoint:
        pnerf.load_state_dict(checkpoint['model_state_dict'], strict=False)

    optimizer = reset_optimizer(stage, pnerf)

    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        pass
    return pnerf, optimizer, start

def train_static(cfg, pnerf, optimizer, start, max_iter, rays_o_all, rays_d_all, viewdirs_all, rgb_all, ray_mask_all):
    pbar = tqdm(range(start, max_iter))
    for i in pbar:
        optimizer.zero_grad()
        global_loss =  pnerf.forward(1, rays_o_all, 
                    rays_d_all, 
                    viewdirs_all, 
                    rgb_all, 
                    rays_mask=ray_mask_all, 
                    H=cfg["H"], W=cfg["W"], bg=1, saving_freq=100, 
                    partial=True, full_batch=False)
        pnerf.backward(1)
        
        optimizer.step()
        pbar.set_description(f"[Training static], loss={global_loss}")

        if i in cfg['pg_scale']:
            pnerf.nerf.scale_volume_grid()
            optimizer = reset_optimizer("static", pnerf)

        if i % 1000 == 0 and i > 0:
            pnerf.entropy_weight *= 2
            pnerf.entropy_weight = min(0.1, pnerf.entropy_weight)

        if i % 100 == 0:
            torch.save({
                'epoch': i+1,
                'model_state_dict': pnerf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'cfg': pnerf.get_kwargs()
            }, os.path.join(cfg["base_dir"], "model/train_static.pt"))

def train_velocity(cfg, pnerf, start, rays_o_all, rays_d_all, viewdirs_all, rgb_all, ray_mask_all, point_gt=None):
    pnerf.dynamic_observer.particle.deactivate_all()
    def evaluate(max_f, optimizer, full_batch):
        optimizer.zero_grad()
        dt = cfg['dt']
        loss = torch.tensor(np.nan, device=device)
        while loss.isnan():
            pnerf.dynamic_observer.simulator.set_dt(dt)
            loss = pnerf.forward(max_f, rays_o_all, 
                    rays_d_all, 
                    viewdirs_all, 
                    rgb_all,
                    ray_mask_all,
                    H=cfg["H"], W=cfg["W"], bg=1, saving_freq=10, partial=False, full_batch=full_batch, point_gt=point_gt)
            dt /= 2
        time_elapsed = time.time() - time0
        print(f"Time elaspsed: {time_elapsed}")
        return loss
        
    if start <= cfg['hit_frame']:
        start = cfg['hit_frame']
        optimizer = reset_optimizer('velocity', pnerf)
        obj = evaluate(cfg['hit_frame'], optimizer, True)
        pnerf.backward(cfg['hit_frame'])
        for _ in range(100):
            options = {"closure": lambda: evaluate(cfg['hit_frame'], optimizer, True), 'current_loss': obj, "c1":1e-6, "c2": 0.99, "max_ls": 20, "ls_debug":True, "interpolate": False}
            obj, grad, lr, d, _, _, _, _, _ = optimizer.step(lambda: pnerf.backward(cfg['hit_frame']), options)
            change = lr * d.abs().max()
            grad_norm = grad.abs().max().item()
            print(f"===== loss: {obj.item()}; param change: {change}; grad_norm: {grad_norm} =====")
            if change < 1e-3:
                break
        
        torch.save({
            'epoch': start+1,
            'model_state_dict': pnerf.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'cfg': pnerf.get_kwargs()
        }, os.path.join(cfg['base_dir'], "model/train_velocity.pt"))

        start = cfg['hit_frame'] + 1

def train_dynamic(cfg, pnerf, optimizer, start, rays_o_all, rays_d_all, viewdirs_all, rgb_all, ray_mask_all, point_gt=None):
    pnerf.dynamic_observer.particle.deactivate_all()
    
    def evaluate(max_f, optimizer, full_batch):
        optimizer.zero_grad()
        loss = torch.tensor(np.nan, device=device)
        dt = cfg['dt']
        while loss.isnan():
            pnerf.dynamic_observer.simulator.set_dt(dt)
            if point_gt is not None:
                loss = pnerf.geoloss_forward(max_f, point_gt)
            else:
                loss = pnerf.forward(max_f, rays_o_all, 
                        rays_d_all, 
                        viewdirs_all, 
                        rgb_all,
                        ray_mask_all,
                        H=cfg["H"], W=cfg["W"], bg=1, saving_freq=10, partial=False, full_batch=full_batch, point_gt=point_gt)
            dt /= 2
        time_elapsed = time.time() - time0
        print(f"Time elaspsed: {time_elapsed}")
        return loss
    
    optimizer = reset_optimizer('dynamic', pnerf)
    for stage, frame_iter in enumerate(zip([cfg["hit_frame"] + cfg["warmup_step"], rgb_all.shape[1]], [50, 100])):
        if stage >= start:
            max_f, iter = frame_iter
            for i in range(iter):
                obj = evaluate(max_f, optimizer, False)
                pnerf.backward(max_f)
                optimizer.step()
                torch.save({
                    'epoch': stage + 1,
                    'model_state_dict': pnerf.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'cfg': pnerf.get_kwargs()
                }, os.path.join(cfg['base_dir'], "model/train_dynamic.pt")) 
        

import argparse
def config_parser():
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True,
                        help='config file path')
    return parser

if __name__=='__main__':
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)['cfg']
    rays_o_all, rays_d_all, viewdirs_all, rgb_all, ray_mask_all = load_data(cfg['data_dir'], cfg['n_cameras'], H=cfg['H'], W=cfg['W'])
    pnerf, optimizer, start = create_model_and_optimizer(cfg, 'static', pnerf=None)
    train_static(cfg, pnerf, optimizer, start, cfg['N_static'], rays_o_all, rays_d_all, viewdirs_all, rgb_all, ray_mask_all)
    
    rgb_all = rgb_all[:, :cfg['n_frames'] , ...]
    pnerf, optimizer, start = create_model_and_optimizer(cfg, 'velocity', pnerf=pnerf)
    train_velocity(cfg, pnerf, start, rays_o_all, rays_d_all, viewdirs_all, rgb_all, ray_mask_all)

    pnerf, optimizer, start = create_model_and_optimizer(cfg, 'dynamic', pnerf=pnerf)
    train_dynamic(cfg, pnerf, optimizer, start, rays_o_all, rays_d_all, viewdirs_all, rgb_all, ray_mask_all)
