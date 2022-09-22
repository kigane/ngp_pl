import os
import warnings

import cv2 as cv
import imageio
import numpy as np
import torch
# optimizer, losses
from einops import rearrange
# models
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
# data
from torch.utils.data import DataLoader
from torchvision import transforms as tf

import models.transfer_net as net
from datasets.style_transfer import SimpleDataset
from models.johnson_net import TransformerNet
from utils import parse_args

warnings.filterwarnings("ignore")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def read_style_image(path, shape):
    img = imageio.imread(path).astype(np.float32)/255.0
    img = cv.resize(img, shape)
    return torch.FloatTensor(img)


class JohnsonSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        # loss
        vgg = net.vgg
        self.vgg19 = nn.Sequential(*list(vgg.children())[:31])
        self.loss_fn = GaytsLoss()

        # model
        self.model = TransformerNet()

        self.style_img = read_style_image(self.hparams.style_image, self.hparams.image_size)

    def forward(self, batch):
        """
        get style transfer result
        """
        pass

    def setup(self, stage):
        """
        setup dataset for each machine
        """
        transform_list = [tf.Resize(self.hparams.image_size),
                          tf.CenterCrop(self.hparams.image_size),
                          tf.ToTensor()]
        if self.hparams.normalize_img:
            transform_list.append(tf.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        transform = tf.Compose(transform_list)
        self.train_dataset = SimpleDataset(self.hparams.content_dir, transform)

    def configure_optimizers(self):
        """
        define optimizers and LR schedulers
        """
        # load_ckpt(self.model, self.hparams.weight_path)
        self.net_opt = Adam(self.model.parameters(), self.hparams.lr)
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)
        return [self.net_opt], [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def training_step(self, batch, batch_idx, *args):
        """
        the complete training loop
        batch: dataloader的内容
        batch_idx: 索引
        """
        results = self(batch, split='train')
        loss_d = self.loss(results, batch)
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        return loss

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = parse_args()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')

    system = JohnsonSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/johnson',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    if hparams.use_wandb:
        logger = WandbLogger(save_dir=f"logs/johnson",
                            project="NeRF",
                            name=hparams.exp_name,
                            log_model=True)  # 最后保存一次模型
    else:
        logger = TensorBoardLogger(save_dir=f"logs/johnson",
                                name=hparams.exp_name,
                                default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                      if hparams.num_gpus > 1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0)

    # if ckpt_path is not none, resume training.
    trainer.fit(system, ckpt_path=hparams.ckpt_path)
