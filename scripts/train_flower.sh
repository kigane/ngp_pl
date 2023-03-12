#!/bin/bash
timer_start=`date "+%Y-%m-%d %H:%M:%S"`

config="--config config/nerf_st/flower.yml"
method="--style_transfer_method adain"
use_filter="--use_guided_filter 0"
style_id=17

python train.py $config $use_filter $method --loop 0 --style_image data/styles/$style_id.jpg --exp_name LLFF_FLOWER_ADAIND0_$style_id
python train.py $config $use_filter $method --loop 1 --style_image data/styles/$style_id.jpg --exp_name LLFF_FLOWER_ADAIND0_$style_id
python train.py $config $use_filter $method --loop 2 --style_image data/styles/$style_id.jpg --exp_name LLFF_FLOWER_ADAIND0_$style_id
python train.py $config $use_filter $method --loop 3 --style_image data/styles/$style_id.jpg --exp_name LLFF_FLOWER_ADAIND0_$style_id
python train.py $config $use_filter $method --loop 4 --style_image data/styles/$style_id.jpg --exp_name LLFF_FLOWER_ADAIND0_$style_id

python train.py $config $use_filter $method --loop 0 --style_image data/styles/11.jpg --exp_name LLFF_FLOWER_ADAIND0_11
python train.py $config $use_filter $method --loop 1 --style_image data/styles/11.jpg --exp_name LLFF_FLOWER_ADAIND0_11
python train.py $config $use_filter $method --loop 2 --style_image data/styles/11.jpg --exp_name LLFF_FLOWER_ADAIND0_11
python train.py $config $use_filter $method --loop 3 --style_image data/styles/11.jpg --exp_name LLFF_FLOWER_ADAIND0_11
python train.py $config $use_filter $method --loop 4 --style_image data/styles/11.jpg --exp_name LLFF_FLOWER_ADAIND0_11

timer_end=`date "+%Y-%m-%d %H:%M:%S"`

duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`

echo "开始： $timer_start"
echo "结束： $timer_end"
echo "耗时： $duration"
