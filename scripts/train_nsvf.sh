#!/bin/bash
set -v
timer_start=`date "+%Y-%m-%d %H:%M:%S"`

style_idx="8"
style="--style_image data/styles/$style_idx.jpg"
method="--style_transfer_method nnst"
use_filter="--use_guided_filter 0"

# exp_drum0="--exp_name NSVF_DRUM_NNST_DEPTH_$style_idx"
# python train.py --config config/nerf_st/nsvf_drum.yml --loop 0 $use_filter $style $exp_drum0 
# python train.py --config config/nerf_st/nsvf_drum.yml --loop 1 $use_filter $style $exp_drum0
# python train.py --config config/nerf_st/nsvf_drum.yml --loop 2 $use_filter $style $exp_drum0
# python train.py --config config/nerf_st/nsvf_drum.yml --loop 3 $use_filter $style $exp_drum0
# python train.py --config config/nerf_st/nsvf_drum.yml --loop 4 $use_filter $style $exp_drum0

# exp_ficus0="--exp_name NSVF_FICUS_NNST_DEPTH_$style_idx"
# python train.py --config config/nerf_st/nsvf_ficus.yml --loop 0 $use_filter $style $exp_ficus0
# python train.py --config config/nerf_st/nsvf_ficus.yml --loop 1 $use_filter $style $exp_ficus0
# python train.py --config config/nerf_st/nsvf_ficus.yml --loop 2 $use_filter $style $exp_ficus0
# python train.py --config config/nerf_st/nsvf_ficus.yml --loop 3 $use_filter $style $exp_ficus0
# python train.py --config config/nerf_st/nsvf_ficus.yml --loop 4 $use_filter $style $exp_ficus0

exp_hotdog0="--exp_name NSVF_HOTDOG_NNST_DEPTH_$style_idx"
python train.py --config config/nerf_st/nsvf_hotdog.yml --loop 0 $method $use_filter $style $exp_hotdog0
python train.py --config config/nerf_st/nsvf_hotdog.yml --loop 1 $method $use_filter $style $exp_hotdog0
python train.py --config config/nerf_st/nsvf_hotdog.yml --loop 2 $method $use_filter $style $exp_hotdog0
python train.py --config config/nerf_st/nsvf_hotdog.yml --loop 3 $method $use_filter $style $exp_hotdog0
python train.py --config config/nerf_st/nsvf_hotdog.yml --loop 4 $method $use_filter $style $exp_hotdog0

exp_mic0="--exp_name NSVF_MIC_NNST_DEPTH_$style_idx"
python train.py --config config/nerf_st/nsvf_mic.yml --loop 0 $method $use_filter $style $exp_mic0
python train.py --config config/nerf_st/nsvf_mic.yml --loop 1 $method $use_filter $style $exp_mic0
python train.py --config config/nerf_st/nsvf_mic.yml --loop 2 $method $use_filter $style $exp_mic0
python train.py --config config/nerf_st/nsvf_mic.yml --loop 3 $method $use_filter $style $exp_mic0
python train.py --config config/nerf_st/nsvf_mic.yml --loop 4 $method $use_filter $style $exp_mic0

# exp_ship0="--exp_name NSVF_SHIP_NNST_DEPTH_$style_idx"
# python train.py --config config/nerf_st/nsvf_ship.yml --loop 0 $use_filter $style $exp_ship0
# python train.py --config config/nerf_st/nsvf_ship.yml --loop 1 $use_filter $style $exp_ship0
# python train.py --config config/nerf_st/nsvf_ship.yml --loop 2 $use_filter $style $exp_ship0
# python train.py --config config/nerf_st/nsvf_ship.yml --loop 3 $use_filter $style $exp_ship0
# python train.py --config config/nerf_st/nsvf_ship.yml --loop 4 $use_filter $style $exp_ship0


# exp_drum="--exp_name NSVF_DRUM_CCPL_DEPTH_GUIDE_$style_idx"
# python train.py --config config/nerf_st/nsvf_drum.yml --loop 0 $use_filter $method $style $exp_drum
# python train.py --config config/nerf_st/nsvf_drum.yml --loop 1 $use_filter $method $style $exp_drum
# python train.py --config config/nerf_st/nsvf_drum.yml --loop 2 $use_filter $method $style $exp_drum
# python train.py --config config/nerf_st/nsvf_drum.yml --loop 3 $use_filter $method $style $exp_drum

# exp_ficus="--exp_name NSVF_FICUS_CCPL_DEPTH_GUIDE_$style_idx"
# python train.py --config config/nerf_st/nsvf_ficus.yml --loop 0 $use_filter $method $style $exp_ficus
# python train.py --config config/nerf_st/nsvf_ficus.yml --loop 1 $use_filter $method $style $exp_ficus
# python train.py --config config/nerf_st/nsvf_ficus.yml --loop 2 $use_filter $method $style $exp_ficus
# python train.py --config config/nerf_st/nsvf_ficus.yml --loop 3 $use_filter $method $style $exp_ficus

# exp_hotdog="--exp_name NSVF_HOTDOG_CCPL_DEPTH_GUIDE_$style_idx"
# python train.py --config config/nerf_st/nsvf_hotdog.yml --loop 0 $use_filter $method $style $exp_hotdog
# python train.py --config config/nerf_st/nsvf_hotdog.yml --loop 1 $use_filter $method $style $exp_hotdog
# python train.py --config config/nerf_st/nsvf_hotdog.yml --loop 2 $use_filter $method $style $exp_hotdog
# python train.py --config config/nerf_st/nsvf_hotdog.yml --loop 3 $use_filter $method $style $exp_hotdog

# exp_mic="--exp_name NSVF_MIC_CCPL_DEPTH_GUIDE_$style_idx"
# python train.py --config config/nerf_st/nsvf_mic.yml --loop 0 $use_filter $method $style $exp_mic
# python train.py --config config/nerf_st/nsvf_mic.yml --loop 1 $use_filter $method $style $exp_mic
# python train.py --config config/nerf_st/nsvf_mic.yml --loop 2 $use_filter $method $style $exp_mic
# python train.py --config config/nerf_st/nsvf_mic.yml --loop 3 $use_filter $method $style $exp_mic

# exp_ship="--exp_name NSVF_SHIP_CCPL_DEPTH_GUIDE_$style_idx"
# python train.py --config config/nerf_st/nsvf_ship.yml --loop 0 $use_filter $method $style $exp_ship
# python train.py --config config/nerf_st/nsvf_ship.yml --loop 1 $use_filter $method $style $exp_ship
# python train.py --config config/nerf_st/nsvf_ship.yml --loop 2 $use_filter $method $style $exp_ship
# python train.py --config config/nerf_st/nsvf_ship.yml --loop 3 $use_filter $method $style $exp_ship

timer_end=`date "+%Y-%m-%d %H:%M:%S"`

duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`

echo "开始： $timer_start"
echo "结束： $timer_end"
echo "耗时： $duration"