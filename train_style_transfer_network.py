from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
# models
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import models.transfer_net as tnet
# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
# data
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from utils import parse_args, train_transform
from icecream import ic, install

install()
ic.configureOutput(prefix=lambda: datetime.now().strftime('%y-%m-%d %H:%M:%S | '),
                   includeContext=True)

import warnings; warnings.filterwarnings("ignore")


class FlatFolderDataset(Dataset):
    def __init__(self, content_dir, style_dir, transform):
        super(FlatFolderDataset, self).__init__()
        self.content_paths = list(Path(content_dir).glob('*'))
        self.style_paths = list(Path(style_dir).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        content = self.content_paths[index]
        style = self.style_paths[index]
        img_c = Image.open(str(content)).convert('RGB')
        img_s = Image.open(str(style)).convert('RGB')
        img_c = self.transform(img_c)
        img_s = self.transform(img_s)
        return img_c, img_s

    def __len__(self):
        return len(self.content_paths)

    def name(self):
        return 'FlatFolderDataset'


class StyleTransferSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        decoder = tnet.decoder
        vgg = tnet.vgg

        vgg.load_state_dict(torch.load(hparams.vgg_pretrained))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        self.model = tnet.StyleTransferNet(vgg, decoder)

    def forward(self, batch):
        Ic, Is = batch
        loss_c, loss_s = self.model(Ic, Is)
        return loss_c, loss_s

    def setup(self, stage):
        """
        setup dataset for each machine
        """
        tf = train_transform()
        self.train_dataset = FlatFolderDataset(self.hparams.content_dir, self.hparams.style_dir, tf)

    def configure_optimizers(self):
        """
        define optimizers and LR schedulers
        """
        opts = []
        self.net_opt = Adam(self.model.decoder.parameters(), self.hparams.lr)
        opts += [self.net_opt]
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

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
        loss_c, loss_s = self(batch)
        loss = loss_c * self.hparams.content_weight + loss_s * self.hparams.style_weight
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss_c', loss_c, True)
        self.log('train/loss_s', loss_s, True)
        self.log('train/loss', loss, True)

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
    
    system = StyleTransferSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/depth_adain',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]
    
    if hparams.use_wandb:
        logger = WandbLogger(save_dir=f"logs/depth_adain",
                             project="NeRF",
                             name=hparams.exp_name,
                             log_model=True)  # 最后保存一次模型
    else:
        logger = TensorBoardLogger(save_dir=f"logs/depth_adain",
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
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=32)

    trainer.fit(system, ckpt_path=hparams.ckpt_path) # if ckpt_path is not none, resume training.


