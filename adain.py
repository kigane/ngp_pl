import imageio
import numpy as np
import torch
from tqdm import tqdm
# optimizer, losses
# models
from pytorch_lightning import LightningModule
# pytorch-lightning
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
import models.transfer_net as Net

import warnings; warnings.filterwarnings("ignore")


class StyleTransferSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        vgg, decoder = Net.vgg, Net.decoder
        vgg.load_state_dict(torch.load(hparams.vgg_ckpt))
        if hparams.decoder_ckpt is not None:
            decoder.load_state_dict(torch.load(hparams.decoder_ckpt))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        self.model = Net.StyleTransferNet(vgg, decoder)
        
    def setup(self, stage):
        """
        setup dataset for each machine
        """
        ret_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/{self.hparams.loop}'
        dataset = dataset_dict['nerf_ret']
        self.train_dataset = dataset(ret_dir)
    
    def configure_optimizers(self):
        """
        define optimizers and LR schedulers
        """   
        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr_st, eps=1e-15)
        opts += [self.net_opt]
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs_st,
                                    self.hparams.lr_st/30)
        return opts, [net_sch]
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=self.hparams.batch_size_st,
                          pin_memory=True)
    
    def training_step(self, batch, batch_idx, *args):
        """
        the complete training loop
        batch: dataloader的内容
        batch_idx: 索引
        """
        content_images = batch
        style_image = imageio.imread(self.hparams.style_img).astype(np.float32)/255.0
        style_image = torch.FloatTensor(style_image)
        loss_c, loss_s = self.model(content_images, style_image)
        loss_c = self.hparams.content_weight * loss_c
        loss_s = self.hparams.style_weight * loss_s
        loss = loss_c + loss_s
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/loss_c', loss_c, True)
        self.log('train/loss_s', loss_s, True)
        return loss

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
