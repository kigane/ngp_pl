#!/bin/bash
timer_start=`date "+%Y-%m-%d %H:%M:%S"`
config="--config config/nerf_st/fern.yml"
python train.py $config --loop 0 --style_image data/styles/8.jpg --exp_name LLFF_FERN_ST_NNST_8
python train.py $config --loop 1 --style_image data/styles/8.jpg --exp_name LLFF_FERN_ST_NNST_8
python train.py $config --loop 2 --style_image data/styles/8.jpg --exp_name LLFF_FERN_ST_NNST_8
python train.py $config --loop 3 --style_image data/styles/8.jpg --exp_name LLFF_FERN_ST_NNST_8
python train.py $config --loop 4 --style_image data/styles/8.jpg --exp_name LLFF_FERN_ST_NNST_8

python train.py $config --loop 0 --style_image data/styles/122.jpg --exp_name LLFF_FERN_ST_NNST_122
python train.py $config --loop 1 --style_image data/styles/122.jpg --exp_name LLFF_FERN_ST_NNST_122
python train.py $config --loop 2 --style_image data/styles/122.jpg --exp_name LLFF_FERN_ST_NNST_122
python train.py $config --loop 3 --style_image data/styles/122.jpg --exp_name LLFF_FERN_ST_NNST_122
python train.py $config --loop 4 --style_image data/styles/122.jpg --exp_name LLFF_FERN_ST_NNST_122

python train.py $config --loop 0 --style_image data/styles/19.jpg --exp_name LLFF_FERN_ST_NNST_19
python train.py $config --loop 1 --style_image data/styles/19.jpg --exp_name LLFF_FERN_ST_NNST_19
python train.py $config --loop 2 --style_image data/styles/19.jpg --exp_name LLFF_FERN_ST_NNST_19
python train.py $config --loop 3 --style_image data/styles/19.jpg --exp_name LLFF_FERN_ST_NNST_19
python train.py $config --loop 4 --style_image data/styles/19.jpg --exp_name LLFF_FERN_ST_NNST_19

timer_end=`date "+%Y-%m-%d %H:%M:%S"`

duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`

echo "开始： $timer_start"
echo "结束： $timer_end"
echo "耗时： $duration"
