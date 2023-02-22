from argparse import Namespace
from datetime import datetime
from glob import glob
from icecream import ic, install
import models.adain_net as tnet
import torch
import torch.nn as nn
import os
import imageio
import utils
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity as ssim

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # 关闭warning
warnings.filterwarnings("ignore", category=UserWarning) # 关闭warning

install()
ic.configureOutput(prefix=lambda: datetime.now().strftime('%y-%m-%d %H:%M:%S | '),
       includeContext=False)

def test_encoder(model):
    x = torch.randn((1, 3, 224, 224))
    out = model(x)
    for y in out:
        print(y.shape)

def show_ckpt():
    slimed = "ckpts/nerf/LEGO_ST_dev00/epoch=2_slim.ckpt"
    raw = "ckpts/nerf/LEGO_ST_dev00/epoch=2.ckpt"
    ic(torch.load(slimed))
    ic(torch.load(raw))
    

def save_video(name, img_paths):
    imageio.mimsave(name, [imageio.imread(img) for img in img_paths],
                        fps=30, macro_block_size=1)
    

def test_img_guided_filter():
    img = cv2.imread("results/colmap/LLFF_FLOWER_ST_PAMA_L1_14/0/000_s_14.jpg", flags=1)
    imgGuide = cv2.imread("results/colmap/LLFF_FLOWER_ST_PAMA_L1_14/0/000.png", flags=1)  # 引导图片

    imgBiFilter = cv2.bilateralFilter(img, d=0, sigmaColor=50, sigmaSpace=10)
    imgGuidedFilter = cv2.ximgproc.guidedFilter(imgGuide, img, 3, 0.1, -1)
    ic((img-imgGuidedFilter).mean())
    ic(img.shape)
    ic(ssim(img, imgGuidedFilter, win_size=7, channel_axis=2))
    ic(ssim(img, imgBiFilter, win_size=7, channel_axis=2))
    ic(ssim(img, imgGuide, win_size=7, channel_axis=2))
    ic(ssim(imgGuide, imgGuidedFilter, win_size=7, channel_axis=2))
    ic(ssim(imgGuide, imgBiFilter, win_size=7, channel_axis=2))

    plt.figure(figsize=(12, 6))
    plt.subplot(141), plt.axis('off'), plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.subplot(142), plt.axis('off'), plt.title("Guide")
    plt.imshow(cv2.cvtColor(imgGuide, cv2.COLOR_BGR2RGB))
    plt.subplot(143), plt.axis('off'), plt.title("cv2.bilateralFilter")
    plt.imshow(cv2.cvtColor(imgBiFilter, cv2.COLOR_BGR2RGB))
    plt.subplot(144), plt.axis('off'), plt.title("cv2.guidedFilter")
    plt.imshow(cv2.cvtColor(imgGuidedFilter, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':    
    print('\033[32m#################################################\033[0m')
    # ckpt_dict = torch.load("ckpts/depth_adain/epoch=4.ckpt")
    # ic(ckpt_dict.keys())
    # keys_to_pop = []
    # for k in ckpt_dict['state_dict']:
    #     if k.startswith('midas'):
    #         keys_to_pop += [k]
    # for k in keys_to_pop:
    #     ckpt_dict['state_dict'].pop(k, None)
    # ic(ckpt_dict['state_dict'].keys())
    
    # separate alpha channel
    # img_path = "data/NSVF/Synthetic_NeRF/Chair/rgb/0_0001.png"
    # img_path = "data/NSVF/Synthetic_NeRF/Lego/rgb/0_0000.png"
    # img = cv2.imread(img_path, flags=cv2.IMREAD_UNCHANGED)
    # img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    # img_alpha = img[:, :, -1] > 0
    # ic(img.shape)
    # ic((img_alpha > 0).sum())
    # ic((img_alpha == 0).sum())
    # plt.figure(figsize=(6, 6))
    # plt.subplot(121), plt.axis('off'), plt.title("RGBA")
    # plt.imshow(img)
    # plt.subplot(122), plt.axis('off'), plt.title("A")
    # plt.imshow(img_alpha, cmap='gray')
    # plt.tight_layout()
    # plt.show()   
    
    hparams = Namespace()
    hparams.dataset_name = 'nsvf'
    hparams.exp_name = 'NSVF_Lego_PAMA_DEPTH_14'
    hparams.loop = 3
    hparams.use_guided_filter = 1
    hparams.fps = 8
    hparams.image_wh = (800, 800)
    utils.save_video(hparams)
    # utils.save_compare_video(hparams)
    
    print('\033[32m#################################################\033[0m')
