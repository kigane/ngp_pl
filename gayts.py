from ast import Raise
import os

import numpy as np
import torch
import utils

from torch.autograd import Variable
from torch.optim import LBFGS, Adam
import models.transfer_net as tnet


def neural_style_transfer(content_img_path, style_img_path, hparams):
    dump_path = os.path.dirname(content_img_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img = utils.prepare_img(content_img_path, hparams.height, device)
    style_img = utils.prepare_img(style_img_path, hparams.height, device)

    if hparams.init_method == 'random':
        gaussian_noise_img = np.random.normal(loc=0, scale=90., size=content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)
    elif hparams.init_method == 'content':
        init_img = content_img
    elif hparams.init_method == 'style':
        style_img_resized = utils.prepare_img(style_img_path, np.asarray(content_img.shape[2:]), device)
        init_img = style_img_resized
    else:
        raise NotImplementedError(f"this init method is not supported! {hparams.init_method}")

    # we are tuning optimizing_img's pixels! (that's why requires_grad=True)
    optimizing_img = Variable(init_img, requires_grad=True)

    encoder = tnet.VGGEncoder(hparams.vgg_ckpt_path)

    #! 内容图像和风格图像分别喂给VGG，并返回特征图[relu1_1,relu2_1,relu3_1,relu4_1]
    content_img_set_of_feature_maps = encoder(content_img)
    style_img_set_of_feature_maps = encoder(style_img)

    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]
    
    num_of_iterations = {
        "lbfgs": 1000,
        "adam": 3000,
    }

    #
    # Start of optimization procedure
    #
    if hparams.optimizer == 'adam':
        optimizer = Adam((optimizing_img,), lr=1e1)
        tuning_step = make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        for cnt in range(num_of_iterations[hparams.optimizer]):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)
    elif hparams.optimizer == 'lbfgs':
        # line_search_fn does not seem to have significant impact on result
        optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations['lbfgs'], line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations[config['optimizer']], should_display=False)

            cnt += 1
            return total_loss

        optimizer.step(closure)

    return dump_path