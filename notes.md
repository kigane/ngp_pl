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

## NGP_pl Code
model.density_grid 
model.grid_coords
这两个有什么用?slim时会被删除。

## 实验记录
使用SNeRF提出的框架。
在不固定NGP的density grid和density net时，训练和推理时需要的采样的个数明显增加了。
说明这样训练之后，density的分布明显更分散了。可能是因为初始训练轮次后的训练数据本身不具有多视角一致性导致的。而且效果很差，可以明显看到有很多floaters。

只固定density grid和Encoding，有一些floaters，且很模糊，效果不如双固定。

