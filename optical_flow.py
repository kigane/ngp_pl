import torch
import torch.nn as nn
from torchvision.io import read_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
import torchvision.transforms.functional as F
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings('ignore')

weights = Raft_Large_Weights.DEFAULT
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


def warp(x, flo):
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

    x = x
    grid = grid.requires_grad_(True)
    vgrid = grid + flo # B,2,H,W
    #图二的每个像素坐标加上它的光流即为该像素点对应在图一的坐标

    # scale grid to [-1,1] 
    ##2019 code
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone()/max(W-1,1)-1.0 
    #取出光流v这个维度，原来范围是0~W-1，再除以W-1，范围是0~1，再乘以2，范围是0~2，再-1，范围是-1~1
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone()/max(H-1,1)-1.0 #取出光流u这个维度，同上

    vgrid = vgrid.permute(0,2,3,1)#from B,2,H,W -> B,H,W,2，为什么要这么变呢？是因为要配合grid_sample这个函数的使用
    output = nn.functional.grid_sample(x, vgrid,align_corners=True)
    """
    对于output中的每一个像素(x, y)，它会根据流值在input中找到对应的像素点(x+u, y+v)，并赋予自己对应点的像素值，这便完成了warp操作。但这个对应点的坐标不一定是整数值，因此要用到插值或者使用邻近值，也就是选项mode的作用。

    那么如何找到对应像素点呢？关键的过程在于grid，若grid(x,y)的两个通道值为(m, n)，则表明output(x,y)的对应点在input的(m, n)处。但这里一般会将m和n的取值范围归一化到[-1, 1]之间，[-1, -1]表示input左上角的像素的坐标，[1, 1]表示input右下角的像素的坐标，对于超出这个范围的坐标，函数将会根据参数padding_mode的设定进行不同的处理。
    """
    mask = torch.ones(x.size())
    mask = nn.functional.grid_sample(mask, vgrid,align_corners=True)

    ##2019 author
    mask[mask<0.9999] = 0
    mask[mask>0] = 1

    ##2019 code
    # mask = torch.floor(torch.clamp(mask, 0 ,1))

    return output*mask


frames, _, _ = read_video('./basketball.mp4')
# (N,H,W,C) --> (N,C,H,W)
frames = frames.permute(0, 3, 1, 2)

# 构造模型输入(N,C,H,W)
img1_batch = torch.stack([frames[100], frames[150]])
img2_batch = torch.stack([frames[101], frames[151]])

# 图片预处理：值域映射到[-1,1]和resize
img1_batch, img2_batch = preprocess(img1_batch, img2_batch)

# 加载模型和权重
device = "cuda" if torch.cuda.is_available() else "cpu"
model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

# 预测光流：返回一系列(12个，递归神经网络模型的迭代次数)迭代预测的光流值，最后一个是最准确的。
list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
predicted_flows = list_of_flows[-1] # (N,2,H,W)
# 注意，预测的光流值单位是像素，没有被归一化。

# 光流可视化
flow_imgs = flow_to_image(predicted_flows)
# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]

imgs = warp(img2_batch, predicted_flows)
img2_batch = [(img1 + 1) / 2 for img1 in img2_batch]
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]
grid = [[a, b, c] for (a, b, c) in zip(img1_batch, img2_batch, imgs)]
plot(grid)
plt.show()


# 几个问题
# 为什么输入img2和前向光流可以预测img_1，不应该是反过来的吗？
# 没有对应的点的颜色是怎么确定的？
# occlusion mask是怎么计算的，表示什么？