## 配置文件
- basic: 默认为config/basic.yml。通用配置。
- config: 训练时指定，不同实验配置。
- loop: 当前为第loop次迭代。用于区分初始nerf训练和后续联合style_transfer的训练过程。-1表示纯NeRF训练。
- 以下可以在配置文件中设置，但命令行中优先级最高。
- style_image： data/styles目录下的文件序号。
- exp_name： 实验名称。
- use_guided_filter： 是否使用引导滤波。
- style_transfer_method: 使用的风格迁移方法。

## 数据加载
测试集的 'img_path': self.img_paths[idx] 是为了保存原图像的mask，供后续迭代过程使用。
定义在base.py中，但是是在具体的子类的read_meta方法中赋值。

## 训练脚本写法
style_idx="122"
style="--style_image data/styles/$style_idx.jpg"
method="--style_transfer_method pama"
use_filter="--use_guided_filter 1"

exp_drum0="--exp_name NSVF_DRUM_PAMA_DEPTH_GUIDE_$style_idx"
python train.py --config config/nerf_st/nsvf_drum.yml --loop 0 $use_filter $style $exp_drum0 
python train.py --config config/nerf_st/nsvf_drum.yml --loop 1 $use_filter $style $exp_drum0
python train.py --config config/nerf_st/nsvf_drum.yml --loop 2 $use_filter $style $exp_drum0
python train.py --config config/nerf_st/nsvf_drum.yml --loop 3 $use_filter $style $exp_drum0
