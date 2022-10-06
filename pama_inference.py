import os
from datetime import datetime
import torch
from torchvision.utils import save_image
from PIL import Image, ImageFile
from utils import parse_args
from models.pama_net import PAMANet
from icecream import ic
from utils import test_transform

Image.MAX_IMAGE_PIXELS = None  
ImageFile.LOAD_TRUNCATED_IMAGES = True
ic.configureOutput(prefix=lambda: datetime.now().strftime('%y-%m-%d %H:%M:%S | '),
                   includeContext=True)

def pama_infer_one_image(content, style, hparams):
    DEVICE = 'cuda'
    
    model = PAMANet(hparams)
    model.eval()
    model.to(DEVICE)
    
    tf = test_transform(*hparams.image_wh)
    
    Ic = tf(Image.open(content).convert("RGB")).to(DEVICE)
    Is = tf(Image.open(style).convert("RGB")).to(DEVICE)
    Ic = Ic.unsqueeze(dim=0)
    Is = Is.unsqueeze(dim=0)
    
    with torch.no_grad():
        Ics = model(Ic, Is)
    # ic(Ics.shape) #? 为啥图片会变大一点啊。 由下采样的时候宽高除不尽导致。

    output_dir = hparams.out_dir
    output_name = f"{output_dir}/{os.path.basename(content).split('.')[0]}_s_{os.path.basename(style).split('.')[0]}{hparams.save_ext}"
    save_image(Ics[0], output_name)
        
        
if __name__ == '__main__':
    hparams = parse_args()
    hparams.out_dir = 'data/lego/train'
    hparams.save_ext = '.jpg'
    pama_infer_one_image(hparams.content, hparams.style, hparams)