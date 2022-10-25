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
    

if __name__ == '__main__':
    # hparams = utils.parse_args()
    # ic(tnet.vgg)
    # ic(tnet.vgg_layers_name_index_map)
    
    # model = tnet.VGGEncoder("data/pretrained/vgg_normalised.pth")
    # test_encoder(model)
    
    # show_ckpt()
    # ic(hparams)
    
    # hparams.loop = 1
    # utils.save_video(hparams)
    
    # img_dir = 'results/nerf/LEGO_ST/0'
    # video_name = 'v.mp4'
    # img_paths = [img for img in sorted(glob(os.path.join(img_dir, '*.jpg'))) if '_s_' in img]
    # # ic(img_paths)
    # save_video(img_dir+"/"+video_name, img_paths)
    # ic("finishsed!")
    # img = Image.open("data/styles/7.jpg")
    # ic(img.size)
    
    print('\033[32m#################################################\033[0m')
    ckpt_dict = torch.load("ckpts/depth_adain/epoch=4.ckpt")
    ic(ckpt_dict.keys())
    keys_to_pop = []
    for k in ckpt_dict['state_dict']:
        if k.startswith('midas'):
            keys_to_pop += [k]
    for k in keys_to_pop:
        ckpt_dict['state_dict'].pop(k, None)
    ic(ckpt_dict['state_dict'].keys())
    