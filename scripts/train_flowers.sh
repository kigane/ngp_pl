#!/bin/bash
set -v
timer_start=`date "+%Y-%m-%d %H:%M:%S"`

style_idx="17"
style="--style_image data/styles/$style_idx.jpg"
method="--style_transfer_method pama"
use_filter="--use_guided_filter 1"

python train.py --config config/nerf_st/flower.yml --loop 0
python train.py --config config/nerf_st/flower.yml --loop 1
python train.py --config config/nerf_st/flower.yml --loop 2
python train.py --config config/nerf_st/flower.yml --loop 3
python train.py --config config/nerf_st/flower.yml --loop 4

python train.py --config config/nerf_st/flower.yml --loop 0 --style_image data/styles/20.jpg --exp_name LLFF_FLOWER_ST_PAMA_20
python train.py --config config/nerf_st/flower.yml --loop 1 --style_image data/styles/20.jpg --exp_name LLFF_FLOWER_ST_PAMA_20
python train.py --config config/nerf_st/flower.yml --loop 2 --style_image data/styles/20.jpg --exp_name LLFF_FLOWER_ST_PAMA_20
python train.py --config config/nerf_st/flower.yml --loop 3 --style_image data/styles/20.jpg --exp_name LLFF_FLOWER_ST_PAMA_20
python train.py --config config/nerf_st/flower.yml --loop 4 --style_image data/styles/20.jpg --exp_name LLFF_FLOWER_ST_PAMA_20

python train.py --config config/nerf_st/flower.yml --loop 0 --style_image data/styles/130.jpg --exp_name LLFF_FLOWER_ST_PAMA_130
python train.py --config config/nerf_st/flower.yml --loop 1 --style_image data/styles/130.jpg --exp_name LLFF_FLOWER_ST_PAMA_130
python train.py --config config/nerf_st/flower.yml --loop 2 --style_image data/styles/130.jpg --exp_name LLFF_FLOWER_ST_PAMA_130
python train.py --config config/nerf_st/flower.yml --loop 3 --style_image data/styles/130.jpg --exp_name LLFF_FLOWER_ST_PAMA_130
python train.py --config config/nerf_st/flower.yml --loop 4 --style_image data/styles/130.jpg --exp_name LLFF_FLOWER_ST_PAMA_130

python train.py --config config/nerf_st/flower.yml --loop 0 --style_image data/styles/8.jpg --exp_name LLFF_FLOWER_ST_PAMA_8
python train.py --config config/nerf_st/flower.yml --loop 1 --style_image data/styles/8.jpg --exp_name LLFF_FLOWER_ST_PAMA_8
python train.py --config config/nerf_st/flower.yml --loop 2 --style_image data/styles/8.jpg --exp_name LLFF_FLOWER_ST_PAMA_8
python train.py --config config/nerf_st/flower.yml --loop 3 --style_image data/styles/8.jpg --exp_name LLFF_FLOWER_ST_PAMA_8
python train.py --config config/nerf_st/flower.yml --loop 4 --style_image data/styles/8.jpg --exp_name LLFF_FLOWER_ST_PAMA_8

python train.py --config config/nerf_st/flower.yml --loop 0 --style_image data/styles/19.jpg --exp_name LLFF_FLOWER_ST_PAMA_19
python train.py --config config/nerf_st/flower.yml --loop 1 --style_image data/styles/19.jpg --exp_name LLFF_FLOWER_ST_PAMA_19
python train.py --config config/nerf_st/flower.yml --loop 2 --style_image data/styles/19.jpg --exp_name LLFF_FLOWER_ST_PAMA_19
python train.py --config config/nerf_st/flower.yml --loop 3 --style_image data/styles/19.jpg --exp_name LLFF_FLOWER_ST_PAMA_19
python train.py --config config/nerf_st/flower.yml --loop 4 --style_image data/styles/19.jpg --exp_name LLFF_FLOWER_ST_PAMA_19

python train.py --config config/nerf_st/flower.yml --loop 0 --style_image data/styles/17.jpg --exp_name LLFF_FLOWER_ST_PAMA_17
python train.py --config config/nerf_st/flower.yml --loop 1 --style_image data/styles/17.jpg --exp_name LLFF_FLOWER_ST_PAMA_17
python train.py --config config/nerf_st/flower.yml --loop 2 --style_image data/styles/17.jpg --exp_name LLFF_FLOWER_ST_PAMA_17
python train.py --config config/nerf_st/flower.yml --loop 3 --style_image data/styles/17.jpg --exp_name LLFF_FLOWER_ST_PAMA_17
python train.py --config config/nerf_st/flower.yml --loop 4 --style_image data/styles/17.jpg --exp_name LLFF_FLOWER_ST_PAMA_17

timer_end=`date "+%Y-%m-%d %H:%M:%S"`

duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`

echo "开始： $timer_start"
echo "结束： $timer_end"
echo "耗时： $duration"
