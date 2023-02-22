## instant ngp 
### 准备nerf数据--colmap
在数据文件夹下，要处理的图片放在同级的images文件夹下
aabb_scale通常设为1即可。越大要求的显存越大。
- data-folder$ python /media/dl/My\ Passport/NeRF/instant-ngp/scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 1
- instant-ngp$ ./build/testbed --mode nerf --scene data/nerf/zl

## ngp_pl
### nerf训练
python train.py --root_dir <path/to/lego> --exp_name Lego --dataset_name nerf

python train.py --root_dir data/lego --exp_name Lego --dataset_name nerf

python train.py --root_dir data/nerf++/tanks_and_temples/tanks_and_temples/tat_training_Truck --exp_name Truck --downsample 0.5 --no_save_test --num_epochs 20 --batch_size 16384 --lr 2e-2 --eval_lpips

### gui
python show_gui.py --root_dir <path/to/lego> --exp_name Lego --dataset_name nerf --ckpt_path <path/to/ckpt>

python show_gui.py --root_dir data/lego --exp_name Lego --dataset_name nerf --ckpt_path ckpts/nerf/lego/epoch=9_slim.ckpt

## NeRF Style Transfer
python train.py --config [config_yml] --loop 0
python train.py --config [config_yml] --loop 1

python show_gui.py --root_dir data/lego --exp_name LEGO_ST --dataset_name nerf --ckpt_path ckpts/nerf/LEGO_ST/epoch=9_slim.ckpt

## Depth-aware style transfer
python train_style_transfer_network.py --config config/style_transfer/depth_adain.yml

## NGP_pl Code
model.density_grid 
model.grid_coords
这两个有什么用?slim时会被删除。

## 实验记录
使用SNeRF提出的框架。
在不固定NGP的density grid和density net时，训练和推理时需要的采样的个数明显增加了。
说明这样训练之后，density的分布明显更分散了。可能是因为初始训练轮次后的训练数据本身不具有多视角一致性导致的。而且效果很差，可以明显看到有很多floaters。

只固定density grid和Encoding，有一些floaters，且很模糊，效果不如双固定。

use PAMA
固定训练得到的NeRF几何形状更好，不固定训练时在测试位姿风格化结果更好，但只在测试位姿结果好，整体形状实际是很noisy的。(可能是后续训练数据少了)

风格化NeRF可以用上所有原始训练图片，以保证风格化NeRF的质量。

depth loss 确实有点用。
使用后，每一轮估计的深度图都基本可以保持不变。相比之下，不使用时，深度图会有明显变化。
固定desity相关参数也可以使深度图基本不变，但整体效果只有颜色变了，而纹理很少保留。

style transfer and image reconstruction. Info loss.

## 其他想法

1. 一开始先进行内容保留较多的风格迁移，每次循环逐步增加风格化程度。
2. 如果风格化图片已经比较一致了，为什么NeRF的渲染结果没有保留好细节。是否可以试试增加InstanNGP的多尺度层数或提高最低分辨率
3. 迭代过程的意义是什么?对Gayts方法而言，可能是NeRF过程学到了几何先验，让初始图像更符合空间一致性，再通过继续优化提高风格化程度。对快速风格迁移而言，其本质是一个将内容图的特征图的前二阶距和风格图的对齐的过程。那么迭代过程就没有必要了。

## 风格图片
19: 神奈川冲浪里
41：方格,海
73: 水墨风
86：水彩，人
107: 纸片老虎
122:　老虎头
120,8:　素描

ns-process-data images --data lego --output-dir lego_processed