#!/bin/bash

python train.py --config config/nerf_st/lego.yml --loop 0
python train.py --config config/nerf_st/lego.yml --loop 1
python train.py --config config/nerf_st/lego.yml --loop 2
python train.py --config config/nerf_st/lego.yml --loop 3
python train.py --config config/nerf_st/lego.yml --loop 4