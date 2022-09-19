## instant ngp 
### 准备nerf数据--colmap
在数据文件夹下，要处理的图片放在同级的images文件夹下
aabb_scale通常设为1即可。越大要求的显存越大。
- data-folder$ python /media/dl/My\ Passport/NeRF/instant-ngp/scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 1
- instant-ngp$ ./build/testbed --mode nerf --scene data/nerf/zl

## ngp_pl
### nerf训练
python train.py --root_dir <path/to/lego> --exp_name Lego --dataset_name colmap

python train.py --root_dir data/lego --exp_name Lego --dataset_name colmap

### gui
python show_gui.py --root_dir <path/to/lego> --exp_name Lego --dataset_name colmap --ckpt_path <path/to/ckpt>

python show_gui.py --root_dir data/lego --exp_name Lego --dataset_name nerf --ckpt_path ckpts/nerf/lego/epoch=9_slim.ckpt
