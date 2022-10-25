#!/bin/bash

python train.py --config config/nerf_st/flower.yml --loop 0
python train.py --config config/nerf_st/flower.yml --loop 1
python train.py --config config/nerf_st/flower.yml --loop 2
python train.py --config config/nerf_st/flower.yml --loop 3
python train.py --config config/nerf_st/flower.yml --loop 4
