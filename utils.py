import argparse
import os

import cv2
import numpy as np
import torch
import yaml


def parse_args():
    """read args from two config files specified by --basic and --config
       default are config/basic.yml and config/config.yml
    """
    desc = "NeRF Style Transfer"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--basic', type=str,
                        default='config/basic.yml', help='basic options')
    parser.add_argument('--config', type=str,
                        default='config/config.yml', help='specific options')
    parser.add_argument('--loop', type=int,
                        default=-1, help='number of loop(nerf->style_transfer)')
    return check_args(parser.parse_args())


def check_args(args):
    """combine arguments"""
    with open(args.basic, 'r') as f:
        basic_config = yaml.safe_load(f)
    with open(args.config, 'r') as f:
        specific_config = yaml.safe_load(f)
    args_dict = vars(args)
    args_dict.update(basic_config)
    args_dict.update(specific_config)
    return args


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        print(f'* {log_dir} does not exist, creating...')
        os.makedirs(log_dir)
    return log_dir


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path: return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)


def slim_ckpt(ckpt_path, save_poses=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # pop unused parameters
    keys_to_pop = ['directions', 'model.density_grid', 'model.grid_coords']
    if not save_poses: keys_to_pop += ['poses']
    for k in ckpt['state_dict']:
        if k.startswith('val_lpips'):
            keys_to_pop += [k]
    for k in keys_to_pop:
        ckpt['state_dict'].pop(k, None)
    return ckpt['state_dict']
