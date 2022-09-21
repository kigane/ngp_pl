import glob
import os
import warnings

import cv2
import imageio
import numpy as np
import torch
# optimizer, losses
from apex.optimizers import FusedAdam
from einops import rearrange
from icecream import ic
# models
from kornia.utils.grid import create_meshgrid3d
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
# data
from torch.utils.data import DataLoader
# metrics
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays
from losses import NeRFLoss
from models.networks import NGP
from models.rendering import MAX_SAMPLES, render
from opt import get_opts
from utils import depth2img, load_ckpt, parse_args, slim_ckpt

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    hparams = parse_args()
    ic(hparams)
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
