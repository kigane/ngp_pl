import torch
import torch.nn as nn
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from einops import rearrange, repeat
import warnings

warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"


def warp(x, flo, padding_mode='border'):
    """
    warp an image/tensor (im1)to im2, according to the optical flow
    x: [B, C, H, W] (im1)
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
        grid = grid.cuda()
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

    flow_x_du = torch.zeros((N, H, W), device=flow.device)
    flow_x_dv = torch.zeros((N, H, W), device=flow.device)
    flow_y_du = torch.zeros((N, H, W), device=flow.device)
    flow_y_dv = torch.zeros((N, H, W), device=flow.device)
    
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
    occlusion = torch.zeros((fw_flow.shape[0], fw_flow.shape[1], fw_flow.shape[2]), device=fw_flow.device)
    occlusion[mask == 1] = 1

    return occlusion


@torch.no_grad()
def temporal_warp_error(img1, img2):
    """
    img1: (N,C,H,W)
    img2: (N,C,H,W)
    model: RAFT
    """
    weights = Raft_Large_Weights.C_T_SKHT_K_V2
    # 图片预处理：值域映射到[-1,1]
    # transforms = weights.transforms() # image range: [0,1] -> [-1,1]
    # transforms(img1, img2)
    # 加载RAFT模型
    model = raft_large(weights=weights, progress=False).to(device)
    model = model.eval()
    img1 = img1.to(device)
    img2 = img2.to(device)
    # 计算前向光流
    fw_flows = model(img1, img2)
    fw_flow = fw_flows[-1] # (N,2,H,W)
    # 计算后向光流
    bw_flows = model(img2, img1)
    bw_flow = bw_flows[-1] # (N,2,H,W)
    # 计算occlusion (N, H, W)
    fw_occ = detect_occlusion(
                rearrange(bw_flow, 'n c h w -> n h w c', c=2),
                rearrange(fw_flow, 'n c h w -> n h w c', c=2))
    # 用光流预测img1
    warp_img1 = warp(img1, fw_flow)
    # 计算warp_img1和img2的差异
    diff = rearrange(torch.square(warp_img1-img2), 'n c h w -> n h w c') *\
            repeat(1-fw_occ, 'n h w -> n h w c', c=3)
    return diff.mean((1, 2, 3))


@torch.no_grad()
def style_warp_error(img1, img2, simg1, simg2):
    """
    img1->img2 ==> fw_flow
    img2->img1 ==> bw_flow
    fw_occ = detect_occlusion(fw_flow, bw_flow)
    simg1--fw_flow-->warp_simg1
    mse(simg2, wawrp_simg1, mask=1-fw_occ)
    
    img1: (N,C,H,W) [-1,1]
    img2: (N,C,H,W) [-1,1]
    model: RAFT
    """
    weights = Raft_Large_Weights.C_T_SKHT_K_V2
    # 加载RAFT模型
    model = raft_large(weights=weights, progress=False).to(device)
    model = model.eval()
    img1 = img1.to(device)
    img2 = img2.to(device)
    # 计算前向光流
    fw_flows = model(img1, img2)
    fw_flow = fw_flows[-1] # (N,2,H,W)
    # 计算后向光流
    bw_flows = model(img2, img1)
    bw_flow = bw_flows[-1] # (N,2,H,W)
    # 计算occlusion (N, H, W)
    fw_occ = detect_occlusion(
                rearrange(bw_flow, 'n c h w -> n h w c', c=2),
                rearrange(fw_flow, 'n c h w -> n h w c', c=2))
    # 用光流预测simg1
    warp_simg1 = warp(simg1, fw_flow.cpu())
    # 计算warp_simg1和simg2的差异
    diff = rearrange(torch.square(warp_simg1-simg2), 'n c h w -> n h w c') *\
            repeat(1-fw_occ.cpu(), 'n h w -> n h w c', c=3)
    return diff.mean((1, 2, 3))

if __name__ == '__main__':
    from glob import glob
    import imageio
    from icecream import ic
    from PIL import Image
    from torchvision import transforms as tf
    
    img_tf = tf.Compose([
        tf.ToTensor(),
        tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dir = "results/colmap/LLFF_FLOWER_ST_ADAIN_DLOSS/4"
    imgs = glob(dir + "/*")
    contents = sorted([im for im in imgs if im.endswith('png') and '_d.' not in im and '_s_' not in im])
    Ic = [img_tf(Image.open(c).convert('RGB')) for c in contents]
    Ic = torch.stack(Ic)
    
    dir = "results/colmap/LLFF_FLOWER_ST_ADAIN_DLOSS/0"
    imgs = glob(dir + "/*")
    stylized = sorted([im for im in imgs if '_s_' in im])
    # stylized = sorted([im for im in imgs if im.endswith('png') and '_d.' not in im and '_s_' not in im])
    Is = [img_tf(Image.open(s).convert('RGB')) for s in stylized]
    Is = torch.stack(Is)
    
    bsize = 7
    img1_batch = Ic[0:bsize]
    img2_batch = Ic[1:bsize+1]
    simg1_batch = Is[0:bsize]
    simg2_batch = Is[1:bsize+1]
    errs = temporal_warp_error(img1_batch, img2_batch)
    # errs = style_warp_error(img1_batch, img2_batch, simg1_batch, simg2_batch)
    ic(errs, errs.mean())
    