#!/bin/bash

python train.py --config config/nerf_st/lego.yml --loop 0 --style_image data/styles/3.jpg --exp_name LEGO_ST_PAMA_DEPTH_3
python train.py --config config/nerf_st/lego.yml --loop 1 --style_image data/styles/3.jpg --exp_name LEGO_ST_PAMA_DEPTH_3
python train.py --config config/nerf_st/lego.yml --loop 2 --style_image data/styles/3.jpg --exp_name LEGO_ST_PAMA_DEPTH_3
python train.py --config config/nerf_st/lego.yml --loop 3 --style_image data/styles/3.jpg --exp_name LEGO_ST_PAMA_DEPTH_3
python train.py --config config/nerf_st/lego.yml --loop 4 --style_image data/styles/3.jpg --exp_name LEGO_ST_PAMA_DEPTH_3