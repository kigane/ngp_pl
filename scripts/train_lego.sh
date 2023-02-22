#!/bin/bash

python train.py --config config/nerf_st/lego.yml --loop 0 --style_image data/styles/6.jpg --exp_name LEGO_ST_PAMA_6
python train.py --config config/nerf_st/lego.yml --loop 1 --style_image data/styles/6.jpg --exp_name LEGO_ST_PAMA_6
python train.py --config config/nerf_st/lego.yml --loop 2 --style_image data/styles/6.jpg --exp_name LEGO_ST_PAMA_6
python train.py --config config/nerf_st/lego.yml --loop 3 --style_image data/styles/6.jpg --exp_name LEGO_ST_PAMA_6
python train.py --config config/nerf_st/lego.yml --loop 4 --style_image data/styles/6.jpg --exp_name LEGO_ST_PAMA_6