import argparse
from glob import glob
import os

import cv2 as cv
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import imageio

from torchvision import transforms
from constants import IMAGENET_MEAN_255, IMAGENET_STD_NEUTRAL


#------------------------------args------------------------------------------

def parse_args():
    """read args from two config files specified by --basic and --config
       default are config/basic.yml and config/config.yml
    """
    desc = "NeRF Style Transfer"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config', type=str,
                        default='config/config.yml', help='specific options')
    parser.add_argument('--loop', type=int,
                        default=-1, help='number of loop(nerf->style_transfer)')
    return check_args(parser.parse_args())


def check_args(args):
    """combine arguments"""
    with open(args.config, 'r') as f:
        specific_config = yaml.safe_load(f)
    args_dict = vars(args)
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


def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x
    else:
        raise ValueError(f'Expected numpy array got {type(x)}')


def depth2img(depth):
    """
    生成深度热力图
    """
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv.applyColorMap((depth*255).astype(np.uint8),
                                  cv.COLORMAP_TURBO)
    return depth_img


def save_video(hparams):
    save_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/videos'
    check_folder(save_dir)
    ret_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}/{hparams.loop}'
    all_imgs = glob(ret_dir + "/*")
    # _d. 为深度图。 _s_ 为风格迁移后的图。 pose.npy为保存的相应的位姿。
    nerf_render_rets = sorted([img for img in all_imgs if ('_d.' not in img) and ('_s_' not in img) and ('poses' not in img)])
    stylized_rets = sorted([img for img in all_imgs if '_s_' in img])
    combined = []
    for n, s in zip(nerf_render_rets, stylized_rets):
        nerf_img = imageio.imread(n)
        stylized_img = imageio.imread(s)
        # (h, w, c)
        combined.append(np.concatenate([nerf_img, stylized_img], axis=1))
    # imageio.mimsave(os.path.join(save_dir, f'{hparams.loop}.mp4'), [imageio.imread(img) for img in nerf_render_rets], fps=24, macro_block_size=1)
    # imageio.mimsave(os.path.join(save_dir, f's_{hparams.loop}.mp4'), [imageio.imread(img) for img in stylized_rets], fps=24, macro_block_size=1)
    imageio.mimsave(os.path.join(save_dir, f'combined_{hparams.loop}.mp4'), combined, fps=24, macro_block_size=1)

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
    # ic(model_dict)
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    checkpoint_.pop('grid_coords', None)
    checkpoint_.pop('density_grid', None)
    model_dict.update(checkpoint_)
    # ic(checkpoint_.keys())
    model.load_state_dict(model_dict)


def load_ngp_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    """
    ! 希望保留desity_grid
    """
    if not ckpt_path: return
    model_dict = model.state_dict()
    # ic(model_dict)
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    # ic(model_dict)
    model.load_state_dict(model_dict)
    

def slim_ckpt(ckpt_path, save_poses=False, test=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # pop unused parameters
    if test:
        keys_to_pop = ['directions']
    else:
        keys_to_pop = ['directions', 'model.density_grid', 'model.grid_coords']
    if not save_poses: keys_to_pop += ['poses']
    for k in ckpt['state_dict']:
        if k.startswith('val_lpips'):
            keys_to_pop += [k]
    for k in keys_to_pop:
        ckpt['state_dict'].pop(k, None)
    if test:
        return ckpt
    return ckpt['state_dict']

#-----------------------------style transfer---------------------------------

def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + \
           torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

###############
# gayts
###############
def generate_out_img_name(hparams):
    ind = os.path.basename(hparams.content_image).split('.')[0]
    style = os.path.basename(hparams.style_image).split('.')[0]
    return f"{ind}_s_{style}_{hparams.optimizer}{hparams.save_ext}" # _s 作为风格化结果的标志


def save_and_maybe_display(optimizing_img, img_id, hparams, should_display=False):
    saving_freq = hparams.saving_freq
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)  # swap channel from 1st to 3rd position: ch, _, _ -> _, _, chr

    # for saving_freq == -1 save only the final result (otherwise save with frequency saving_freq and save the last pic)
    if img_id == hparams.iterations-1 or (saving_freq > 0 and img_id % saving_freq == 0):
        out_img_name = str(img_id).zfill(4) + ".jpg" if saving_freq != -1 else generate_out_img_name(hparams)
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')
        cv.imwrite(os.path.join(hparams.out_dir, out_img_name), dump_img[:, :, ::-1])

    if should_display:
        plt.imshow(np.uint8(get_uint8_range(out_img)))
        plt.show()
        
####################
# AdaIN
####################
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    """
    preserve color
    """
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f
    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)),
                 source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())
