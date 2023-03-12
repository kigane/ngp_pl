# Core Imports
import time
import argparse
import random
import os

# External Dependency Imports
from imageio import imwrite
import torch
import numpy as np

# Internal Project Imports
from util.vgg import Vgg16Pretrained
from util import misc as misc
from util.misc import load_path_for_pytorch
from util.stylize import produce_stylization
from PIL import Image


def nnst_infer_one_image(content, style, hparams):
    # Define command line parser and get command line arguments
    high_res = False
    no_flip = False
    content_loss = False
    dont_colorize = True
    alpha = 0.75

    # Interpret command line arguments
    content_path = content
    style_path = style
    output_path = f"{hparams.out_dir}/{os.path.basename(content).split('.')[0]}_s_{os.path.basename(style).split('.')[0].split('_')[-1]}{hparams.save_ext}"

    max_scls = 4
    sz = 512
    if high_res:
        max_scls = 5
        sz = 1024
    flip_aug = (not no_flip)
    content_loss = content_loss
    misc.USE_GPU = True
    content_weight = 1. - alpha

    # Error checking for arguments
    # error checking for paths deferred to imageio
    assert (0.0 <= content_weight) and (content_weight <= 1.0), "alpha must be between 0 and 1"
    assert torch.cuda.is_available() or (not misc.USE_GPU), "attempted to use gpu when unavailable"

    # Define feature extractor
    cnn = misc.to_device(Vgg16Pretrained())
    phi = lambda x, y, z: cnn.forward(x, inds=y, concat=z)

    # Load images
    content_im_orig = misc.to_device(load_path_for_pytorch(content_path, target_size=sz)).unsqueeze(0)
    style_im_orig = misc.to_device(load_path_for_pytorch(style_path, target_size=sz)).unsqueeze(0)

    # Run Style Transfer
    torch.cuda.synchronize()
    output = produce_stylization(content_im_orig, style_im_orig, phi,
                                max_iter=200,
                                lr=2e-3,
                                content_weight=content_weight,
                                max_scls=max_scls,
                                flip_aug=flip_aug,
                                content_loss=content_loss,
                                dont_colorize=dont_colorize)
    torch.cuda.synchronize()

    # Convert from pyTorch to numpy, clip to valid range
    new_im_out = np.clip(output[0].permute(1, 2, 0).detach().cpu().numpy(), 0., 1.)

    # Save stylized output
    save_im = (new_im_out * 255).astype(np.uint8)
    save_im = Image.fromarray(save_im).resize((hparams.image_wh[0], hparams.image_wh[1]))
    
    imwrite(output_path, save_im)

    # Free gpu memory in case something else needs it later
    if misc.USE_GPU:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    import utils
    import time

    time_start=time.time()

    hparams = utils.parse_args()
    style_id = 107
    content_lst = [
        'results/001.png',]
        # f'results/001_s_{style_id}.jpg',
        # f'results/001_s_{style_id}_s_{style_id}.jpg',
        # f'results/001_s_{style_id}_s_{style_id}_s_{style_id}.jpg',
        # f'results/001_s_{style_id}_s_{style_id}_s_{style_id}_s_{style_id}.jpg']
    # hparams.content = 'results/001.png'
    hparams.style = f'data/styles/{style_id}.jpg'
    hparams.out_dir = 'results'
    hparams.save_ext = '.jpg'
    hparams.image_wh = (800, 600)
    for c in content_lst:
        hparams.content = c
        nnst_infer_one_image(hparams.content, hparams.style, hparams)

    time_end=time.time()
    print('time cost',time_end-time_start,'s')