import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights

import copy
from einops import repeat, rearrange
import cv2 as cv
from icecream import ic
import os
from utils import generate_out_img_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cnn = models.vgg19(
    weights=VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def image_loader(image_name, loader):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    # normalize
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().requires_grad_(True).view(-1, 1, 1)
        self.std = std.clone().detach().requires_grad_(True).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(
        normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=1000,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    ic('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    ic('Optimizing..')
    run = [0]
    output = input_img.clone().detach()
    best_loss = 1e10
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            nonlocal best_loss
            nonlocal output

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score

            loss.backward()

            run[0] += 1
            if run[0] % 20 == 0:
                ic('Run {}: Style Loss : {:2f} Content Loss: {:2f}'.format(run, 
                    style_score.item(), content_score.item()))
                if style_score + content_score < best_loss:
                    best_loss = style_score + content_score
                    output = input_img.clone().detach().data.clamp_(0, 1)

            return style_score + content_score

        optimizer.step(closure)

    # input_img.data.clamp_(0, 1)
    # return input_img

    return output


def neural_style_transfer(content_img_path, style_img_path, hparams):
    # h, w
    # ic(hparams)
    w, h = hparams.image_wh
    imsize = (h, w)

    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    style_img = image_loader(style_img_path, loader=loader)
    if style_img.shape[1] == 1:
        style_img = repeat(style_img, "n 1 w h -> n c w h", c=3)

    content_img = image_loader(content_img_path, loader=loader)

    input_img = content_img.clone()
    # input_img = torch.randn(content_img.data.size(), device=device)
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img, num_steps=hparams.num_steps)

    out_img_name = generate_out_img_name(hparams)
    out_img_name = os.path.join(hparams.out_dir, out_img_name)
    cv.imwrite(out_img_name, rearrange(output,
               'n c h w -> n h w c').squeeze().detach().cpu().numpy()[:, :, ::-1] * 255.0)


if __name__ == '__main__':
    from utils import parse_args
    import time

    
    hparams = parse_args()
    style_id = 107
    content_lst = [
        'results/001.png',
        f'results/001_s_{style_id}.jpg',
        f'results/001_s_{style_id}_s_{style_id}.jpg',
        f'results/001_s_{style_id}_s_{style_id}_s_{style_id}.jpg',
        f'results/001_s_{style_id}_s_{style_id}_s_{style_id}_s_{style_id}.jpg']
    # hparams.content = 'results/001.png'
    hparams.style_image = f'data/styles/{style_id}.jpg'
    hparams.out_dir = 'results'
    hparams.save_ext = '.jpg'
    hparams.image_wh = (800, 600)
    hparams.num_steps = 1000
    for c in content_lst:
        time_start=time.time()
        hparams.content_image = c
        neural_style_transfer(hparams.content_image, hparams.style_image, hparams)
        time_end=time.time()
        print('time cost',time_end-time_start,'s')
