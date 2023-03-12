from datetime import datetime
from pathlib import Path
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import models.adain_net as tnet
import utils
from tqdm import tqdm
from icecream import ic
ic.configureOutput(prefix=lambda: datetime.now().strftime('%y-%m-%d %H:%M:%S | '),
                   includeContext=False)

def test_transform(size, crop):
    transform_list = []
    if size:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(content_f.device)
        base_feat = utils.adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = utils.adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def style_transfer_one_image(path, style_image, hparams):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = hparams.out_dir
    content_path = Path(path)
    style_path = style_image

    decoder = tnet.decoder
    vgg = tnet.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(hparams.decoder_pretrained))
    vgg.load_state_dict(torch.load(hparams.vgg_pretrained))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(hparams.content_size, hparams.crop)
    style_tf = test_transform(hparams.style_size, hparams.crop)

    content = content_tf(Image.open(str(content_path)))
    style = style_tf(Image.open(str(style_path)).convert("RGB"))
    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)
    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style,
                                hparams.alpha)
    output = output.cpu()
    output_name = f"{output_dir}/{os.path.basename(content_path).split('.')[0]}_s_{os.path.basename(style_path).split('.')[0]}{hparams.save_ext}"
    save_image(output, str(output_name))


if __name__ == '__main__':
    from utils import parse_args
    hparams = parse_args()
    style_id = 107
    content_lst = [
        'results/001.png',
        f'results/001_s_{style_id}.jpg',
        f'results/001_s_{style_id}_s_{style_id}.jpg',
        f'results/001_s_{style_id}_s_{style_id}_s_{style_id}.jpg',
        f'results/001_s_{style_id}_s_{style_id}_s_{style_id}_s_{style_id}.jpg']
    # hparams.content = 'results/001.png'
    hparams.style = f'data/styles/{style_id}.jpg'
    hparams.out_dir = 'results'
    hparams.save_ext = '.jpg'
    hparams.image_wh = (800, 600)
    for c in content_lst:
        hparams.content = c
        style_transfer_one_image(hparams.content, hparams.style, hparams)