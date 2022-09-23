import numpy as np
import torch
import utils

from torch.autograd import Variable
from torch.optim import LBFGS, Adam
import models.transfer_net as tnet


def build_loss(net, optimizing_img, target_representations, hparams):
    target_content_representation, target_style_representation = target_representations

    current_set_of_feature_maps = net(optimizing_img)
    current_content_representation = current_set_of_feature_maps[hparams.content_feature_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)

    style_loss = 0.0
    current_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in hparams.style_features_indices]

    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)

    #! 计算图像的水平，竖直方向的一阶差分的总和
    tv_loss = utils.total_variation(optimizing_img)

    total_loss = hparams.content_weight * content_loss + \
                 hparams.style_weight * style_loss + \
                 hparams.tv_weight * tv_loss

    return total_loss, content_loss, style_loss, tv_loss


def make_tuning_step(neural_net, optimizer, target_representations, hparams):
    # Builds function that performs a step in the tuning loop
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, hparams)
        # Computes gradients
        total_loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss

    # Returns the function that will be called inside the tuning loop
    return tuning_step

def neural_style_transfer(content_img_path, style_img_path, hparams):
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

    # 待优化的图像。优化完即为最终结果图。
    optimizing_img = Variable(init_img, requires_grad=True)

    encoder = tnet.VGGEncoder(hparams.vgg_ckpt_path, hparams.features)

    #! 内容图像和风格图像分别喂给VGG，并返回特征图[relu1_1, relu2_1, relu3_1, conv4_2, relu4_1, relu5_1]
    content_img_set_of_feature_maps = encoder(content_img)
    style_img_set_of_feature_maps = encoder(style_img)

    target_content_representation = content_img_set_of_feature_maps[hparams.content_feature_index].squeeze(axis=0) # (c, h, w)
    target_style_representation = [utils.gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in hparams.style_features_indices] # (1, c, c)
    target_representations = (target_content_representation, target_style_representation)

    #
    # Start of optimization procedure
    #
    if hparams.optimizer == 'adam':
        optimizer = Adam((optimizing_img,), lr=hparams.lr)
        tuning_step = make_tuning_step(encoder, optimizer, target_representations, hparams)
        for cnt in range(hparams.iterations):
            total_loss, content_loss, style_loss, tv_loss = tuning_step(optimizing_img)
            with torch.no_grad():
                print(f'Adam | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={hparams.content_weight * content_loss.item():12.4f}, style loss={hparams.style_weight * style_loss.item():12.4f}, tv loss={hparams.tv_weight * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, cnt, hparams, should_display=False)
    elif hparams.optimizer == 'lbfgs':
        # line_search_fn does not seem to have significant impact on result
        optimizer = LBFGS((optimizing_img,), max_iter=hparams.iterations, line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            if torch.is_grad_enabled():
                optimizer.zero_grad()
            total_loss, content_loss, style_loss, tv_loss = build_loss(encoder, optimizing_img, target_representations, hparams)
            if total_loss.requires_grad:
                total_loss.backward()
            with torch.no_grad():
                print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={hparams.content_weight * content_loss.item():12.4f}, style loss={hparams.style_weight * style_loss.item():12.4f}, tv loss={hparams.tv_weight * tv_loss.item():12.4f}')
                utils.save_and_maybe_display(optimizing_img, cnt, hparams, should_display=False)

            cnt += 1
            return total_loss

        optimizer.step(closure)

if __name__ == '__main__':
    hparams = utils.parse_args()
    neural_style_transfer(hparams.content_image, hparams.style_image, hparams)