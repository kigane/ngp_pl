import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import cv2 as cv
from tqdm import tqdm

import models.ccpl_net as net
from utils import makeVideo, parse_args
from datasets.ccpl_dataset import CCPLDataset

import warnings
warnings.filterwarnings("ignore")

def test_transform(size, crop):
    transform_list = []
    if size:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def transform(vgg, decoder, SCT, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    cF = vgg(content)
    sF = vgg(style)
    
    t = SCT(cF, sF)
    return t

def loadImg(imgPath, size):
    img = Image.open(imgPath).convert('RGB')
    transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()])
    return transform(img)

def ccpl_inference_frames(path, style_image, hparams):
    styleV = loadImg(style_image, hparams.style_size).unsqueeze(0)
    content_dataset = CCPLDataset(path,
                          loadSize = hparams.content_size,
                          fineSize = hparams.content_size,
                          test     = True,
                          video    = True)
    content_loader = torch.utils.data.DataLoader(dataset    = content_dataset,
                                                batch_size = 1,
                                                shuffle    = False)

    decoder = net.decoder
    vgg = net.vgg
    network = net.CCPLNet(vgg, decoder, hparams.testing_mode)
    SCT = network.SCT

    SCT.eval()
    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(hparams.ccpl_decoder))
    vgg.load_state_dict(torch.load(hparams.vgg))
    SCT.load_state_dict(torch.load(hparams.SCT))
    decoder = decoder if hparams.testing_mode == 'art' else nn.Sequential(*list(net.decoder.children())[10:])
    vgg = nn.Sequential(*list(vgg.children())[:31]) if hparams.testing_mode == 'art' else nn.Sequential(*list(vgg.children())[:18])
    
    vgg.cuda()
    decoder.cuda()
    SCT.cuda()
    
    contentV = torch.Tensor(1,3,*hparams.content_size)
    styleV = styleV.cuda()
    contentV = contentV.cuda()
    result_frames = []
    contents = []
    # style = styleV.squeeze(0).cpu().numpy()
    # sF = vgg(styleV)
    print("Transfering")
    for i,(content,contentName) in enumerate(tqdm(content_loader)):
        # print('Transfer frame %d...'%i)
        contentName = contentName[0]
        contentV.resize_(content.size()).copy_(content)
        contents.append(content.squeeze(0).float().numpy())
        # forward
        with torch.no_grad():
            gF = transform(vgg, decoder, SCT, contentV, styleV)
            transfer = decoder(gF)
        transfer = transfer.clamp(0,1)
        result_frames.append(transfer.squeeze(0).cpu().numpy())

    print("Saving")
    h, w = hparams.style_size
    for idx, frame in enumerate(tqdm(result_frames)):
        frame = frame.transpose((1,2,0))
        frame = frame[...,::-1]
        frame = frame * 255
        frame = cv.resize(frame,(w,h))
        cv.imwrite(f"{path}/{idx:03d}_s_{os.path.basename(style_image).split('.')[0]}{hparams.save_ext}", frame)

    
if __name__ == '__main__':
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok = True, parents = True)

    styleV = loadImg(args.style_path, args.style_size).unsqueeze(0)  

    content_dataset = CCPLDataset(args.content_dir,
                            loadSize = args.content_size,
                            fineSize = args.content_size,
                            test     = True,
                            video    = True)
    content_loader = torch.utils.data.DataLoader(dataset    = content_dataset,
                                                batch_size = 1,
                                                shuffle    = False)

    decoder = net.decoder
    vgg = net.vgg
    network = net.CCPLNet(vgg, decoder, args.testing_mode)
    SCT = network.SCT

    SCT.eval()
    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    SCT.load_state_dict(torch.load(args.SCT))
    decoder = decoder if args.testing_mode == 'art' else nn.Sequential(*list(net.decoder.children())[10:])
    vgg = nn.Sequential(*list(vgg.children())[:31]) if args.testing_mode == 'art' else nn.Sequential(*list(vgg.children())[:18])

    vgg.to(device)
    decoder.to(device)
    SCT.to(device)

    contentV = torch.Tensor(1,3,*args.content_size)
    styleV = styleV.cuda()
    contentV = contentV.cuda()
    result_frames = []
    contents = []
    style = styleV.squeeze(0).cpu().numpy()
    sF = vgg(styleV)
    for i,(content,contentName) in enumerate(tqdm(content_loader)):
        # print('Transfer frame %d...'%i)
        contentName = contentName[0]
        contentV.resize_(content.size()).copy_(content)
        contents.append(content.squeeze(0).float().numpy())
        # forward
        with torch.no_grad():
            gF = transform(vgg, decoder, SCT, contentV, styleV)
            transfer = decoder(gF)
        transfer = transfer.clamp(0,1)
        result_frames.append(transfer.squeeze(0).cpu().numpy())

    makeVideo(contents,style,result_frames,args.output)
        
