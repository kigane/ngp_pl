import os
from datetime import datetime
import torch
from torchvision.utils import save_image
from PIL import Image, ImageFile
from utils import parse_args
from models.pama_net import PAMANet
from icecream import ic
from utils import test_transform
import cv2 as cv

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
    # ic(content)
    # if (content.endswith("png")):
    #     Ic_mask = cv.imread(content, cv.IMREAD_UNCHANGED)[:,:,-1] # png alpha
    #     Ic_mask = (Ic_mask > 0).astype(float)
    #     Ic_mask = torch.from_numpy(Ic_mask).unsqueeze(0)
    Is = tf(Image.open(style).convert("RGB")).to(DEVICE)
    Ic = Ic.unsqueeze(dim=0)
    Is = Is.unsqueeze(dim=0)
    
    with torch.no_grad():
        Ics = model(Ic, Is)
    # ic(Ics.shape) #? 为啥图片会变大一点啊。 由下采样的时候宽高除不尽导致。

    output_dir = hparams.out_dir
    output_name = f"{output_dir}/{os.path.basename(content).split('.')[0]}_s_{os.path.basename(style).split('.')[0].split('_')[-1]}{hparams.save_ext}"
    # if (content.endswith("png")): 
    #     ret = Ics[0].cpu() * Ic_mask
    #     ret = torch.concat([ret, Ic_mask], dim=0)
    #     save_image(ret, output_name.replace('.jpg', '.png'))
    # else:
    save_image(Ics[0], output_name)
        
        
if __name__ == '__main__':
    hparams = parse_args()
    hparams.out_dir = 'results'
    hparams.save_ext = '.jpg'
    hparams.image_wh = (800, 600)
    pama_infer_one_image(hparams.content, hparams.style, hparams)