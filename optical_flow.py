import torch
import torch.nn as nn
from torchvision.io import read_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
import torchvision.transforms.functional as F
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt
import numpy as np
import warnings
from einops import rearrange, repeat
from icecream import ic

warnings.filterwarnings('ignore')

weights = Raft_Large_Weights.C_T_SKHT_K_V2
transforms = weights.transforms() # image range: [0,1] -> [-1,1]


plt.rcParams["savefig.bbox"] = "tight"
# sphinx_gallery_thumbnail_number = 2


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()


def preprocess(img1_batch, img2_batch, size=[520, 960]):
    img1_batch = F.resize(img1_batch, size=size)
    img2_batch = F.resize(img2_batch, size=size)
    return transforms(img1_batch, img2_batch)


def warp(x, flo, padding_mode='border'):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    if x.is_cuda:
        grid.cuda()
    vgrid = grid - flo # B,2,H,W 前向预测:frame1+flow=>frame2
    # grid为未移动的坐标，相当于frame1的像素位置。加上正向flow以后为frame2的像素位置
    # vgrid = grid + flo # B,2,H,W 反向预测frame2+flow=>frame1。
    #图二的每个像素坐标加上它的光流即为该像素点对应在图一的坐标

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0 
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1) # from B,2,H,W -> B,H,W,2，为什么要这么变呢？是因为要配合grid_sample这个函数的使用
    output = nn.functional.grid_sample(
        x, vgrid, padding_mode=padding_mode,
        mode='nearest', align_corners=True)
    """
    对于output中的每一个像素(x, y)，它会根据流值在input中找到对应的像素点(x+u, y+v)，并赋予自己对应点的像素值，这便完成了warp操作。但这个对应点的坐标不一定是整数值，因此要用到插值或者使用邻近值，也就是选项mode的作用。

    那么如何找到对应像素点呢？关键的过程在于grid，若grid(x,y)的两个通道值为(m, n)，则表明output(x,y)的对应点在input的(m, n)处。但这里一般会将m和n的取值范围归一化到[-1, 1]之间，[-1, -1]表示input左上角的像素的坐标，[1, 1]表示input右下角的像素的坐标，对于超出这个范围的坐标，函数将会根据参数padding_mode的设定进行不同的处理。

    grid_sample:
    for pixel in output:
        pos = grid(piexl.pos)
        pixel = input(pos) // pos不一定是整数，因此可能需要插值。
    """
    return output


def compute_flow_magnitude(flow):
    """
    flow: N,H,W,2
    """
    flow_mag = flow[:, :, :, 0] ** 2 + flow[:, :, :, 1] ** 2
    return flow_mag


def compute_flow_gradients(flow):
    """
    flow: N,H,W,2
    """
    N = flow.shape[0]
    H = flow.shape[1]
    W = flow.shape[2]

    flow_x_du = torch.zeros((N, H, W))
    flow_x_dv = torch.zeros((N, H, W))
    flow_y_du = torch.zeros((N, H, W))
    flow_y_dv = torch.zeros((N, H, W))
    
    flow_x = flow[:, :, :, 0]
    flow_y = flow[:, :, :, 1]

    flow_x_du[:, :, :-1] = flow_x[:, :, :-1] - flow_x[:, :, 1:]
    flow_x_dv[:, :-1, :] = flow_x[:, :-1, :] - flow_x[:, 1:, :]
    flow_y_du[:, :, :-1] = flow_y[:, :, :-1] - flow_y[:, :, 1:]
    flow_y_dv[:, :-1, :] = flow_y[:, :-1, :] - flow_y[:, 1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv


def detect_occlusion(fw_flow, bw_flow):
    """
    fw_flow: (N,H,W,2)
    bw_flow: (N,H,W,2)
    """
    with torch.no_grad():
        # warp fw-flow to img2
        fw_flow_w = warp(
            rearrange(fw_flow, 'n h w c -> n c h w'),
            rearrange(bw_flow, 'n h w c -> n c h w'),
            ) # (N,2,H,W)
        fw_flow_w = rearrange(fw_flow_w, 'n c h w -> n h w c')

    ## occlusion
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5

    ## motion boundary
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2
    
    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    ## combine mask
    mask = torch.logical_or(mask1, mask2)
    occlusion = torch.zeros((fw_flow.shape[0], fw_flow.shape[1], fw_flow.shape[2]))
    occlusion[mask == 1] = 1

    return occlusion

@torch.no_grad()
def temporal_warp_error(img1, img2, model):
    """
    img1: (N,C,H,W)
    img2: (N,C,H,W)
    model: RAFT
    """
    fw_flows = model(img1, img2)
    fw_flow = fw_flows[-1] # (N,2,H,W)
    bw_flows = model(img2, img1)
    bw_flow = bw_flows[-1] # (N,2,H,W)
    fw_occ = detect_occlusion(
                rearrange(bw_flow, 'n c h w -> n h w c', c=2),
                rearrange(fw_flow, 'n c h w -> n h w c', c=2))
    ic(fw_occ.shape) # (N, H, W)
    warp_img2 = warp(img2, fw_flow)
    diff = rearrange(torch.square(warp_img2-img1), 'n c h w -> n h w c') * repeat(1-fw_occ, 'n h w -> n h w c', c=3)
    ic(diff.mean((1, 2, 3)))
    ic(diff.sum((1, 2, 3)))
    return fw_occ



frames, _, _ = read_video('./basketball.mp4')
# (N,H,W,C) --> (N,C,H,W)
frames = frames.permute(0, 3, 1, 2)

# 构造模型输入(N,C,H,W)
img1_batch = torch.stack([frames[100], frames[150]])
img2_batch = torch.stack([frames[109], frames[159]])

# 图片预处理：值域映射到[-1,1]和resize
img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

# 加载模型和权重
device = "cuda" if torch.cuda.is_available() else "cpu"
model = raft_large(weights=Raft_Large_Weights.C_T_SKHT_K_V2, progress=False).to(device)
model = model.eval()

img1_batch.requires_grad_(False)
img2_batch.requires_grad_(False)
fw_occ = temporal_warp_error(img1_batch, img2_batch, model)

# 预测光流：返回一系列(12个，递归神经网络模型的迭代次数)迭代预测的光流值，最后一个是最准确的。
list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
predicted_flows = list_of_flows[-1] # (N,2,H,W)
# 注意，预测的光流值单位是像素，没有被归一化。

# 光流可视化
flow_imgs = flow_to_image(predicted_flows)
# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]

imgs = warp(img1_batch, predicted_flows)
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
img2_batch = [(img2 + 1) / 2 for img2 in img2_batch]
imgs = [(img + 1) / 2 for img in imgs]
# imgs = [torch.abs(img - img1) for img, img1 in zip(imgs, img2_batch)]
grid = [[a, b, c, d] for (a, b, c, d) in zip(img1_batch, img2_batch, imgs, repeat(fw_occ, 'n h w -> n c h w', c=3))]
plot(grid)
plt.show()


# 几个问题
# 为什么输入img2和前向光流可以预测img_1，不应该是反过来的吗？
# 没有对应的点的颜色是怎么确定的？padding_mode决定
# 为什么会有一大片蓝色？ warp后的图片没有归一化。
# occlusion mask是怎么计算的，表示什么？