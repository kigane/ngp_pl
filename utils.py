import argparse
import os

import cv2 as cv
import numpy as np
import torch
import yaml

from torchvision import transforms
from constants import IMAGENET_MEAN_255, IMAGENET_STD_NEUTRAL


#------------------------------args------------------------------------------

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


#----------------------------image utils-------------------------------------

def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the height
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


def prepare_img(img_path, target_shape, device):
    img = load_image(img_path, target_shape=target_shape)

    # normalize using ImageNet's mean
    # [0, 255] range worked much better for me than [0, 1] range (even though PyTorch models were trained on latter)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = transform(img).to(device).unsqueeze(0)

    return img


def depth2img(depth):
    """
    生成深度热力图
    """
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv.applyColorMap((depth*255).astype(np.uint8),
                                  cv.COLORMAP_TURBO)
    return depth_img

#-------------------------------model----------------------------------------

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
