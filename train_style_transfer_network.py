from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms

# models
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import models.adain_net as tnet
# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import Adam
# data
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None # 取消PIL的最大像素限制

from utils import parse_args, train_transform
from icecream import ic, install

install()
ic.configureOutput(prefix=lambda: datetime.now().strftime('%y-%m-%d %H:%M:%S | '),
                   includeContext=True)

import warnings; warnings.filterwarnings("ignore")


def get_slimmed_ckpt(ckpt_path):
    # 加载pl的checkpoint
    ckpt_dict = torch.load(ckpt_path)
    ckpt = {}
    # 只要model的权重
    for k, v in ckpt_dict['state_dict'].items():
        if not k.startswith('midas'):
            ckpt[k[6:]] = v
    return ckpt


class FlatFolderDataset(Dataset):
    def __init__(self, content_dir, style_dir, transform):
        super(FlatFolderDataset, self).__init__()
        self.content_paths = list(Path(content_dir).glob('*'))
        self.style_paths = list(Path(style_dir).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        content = self.content_paths[index]
        style = self.style_paths[index % len(self.style_paths)]
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
        vgg.load_state_dict(torch.load(self.hparams.vgg_pretrained))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        self.model = tnet.StyleTransferNet(vgg, decoder)
        
        # 深度预测网络
        if hparams.depth_weight != 0:
            midas = torch.hub.load("data/intel-isl_MiDaS_master", "DPT_Large", source='local')
            for param in midas.parameters():
                param.requires_grad = False      
            self.midas = midas   
            self.midas.eval()    

    def forward(self, batch):
        Ic, Is = batch
        loss_c, loss_s, Ics = self.model(Ic, Is)
        loss_d = 0
        if self.hparams.depth_weight != 0:
            Icd = self.midas(Ic)
            Icsd = self.midas(Ics)
            loss_d = self.model.mse_loss(Icd, Icsd)
        return loss_c, loss_s, loss_d

    def setup(self, stage):
        """
        setup dataset for each machine
        """
        tf = transforms.Compose([
            transforms.Resize(self.hparams.image_size),
            transforms.CenterCrop(self.hparams.crop_size),
            transforms.ToTensor(), # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # [-1, 1]
        ])
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
        loss_c, loss_s, loss_d = self(batch)
        loss = loss_c * self.hparams.content_weight + loss_s * self.hparams.style_weight + loss_d * self.hparams.depth_weight
        self.log('lr', self.net_opt.param_groups[0]['lr'], True)
        self.log('train/loss_c', loss_c, True)
        self.log('train/loss_s', loss_s, True)
        self.log('train/loss_d', loss_d, True)
        self.log('loss', loss)

        return loss

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = parse_args()
    # ic(hparams)
    if hparams.ckpt_path:
        decoder = tnet.decoder
        decoder.load_state_dict(torch.load(hparams.decoder))
        vgg = tnet.vgg
        vgg.load_state_dict(torch.load(hparams.vgg_pretrained))
        vgg = nn.Sequential(*list(vgg.children())[:31])
        model = tnet.StyleTransferNet(vgg, decoder)
        model.load_state_dict(get_slimmed_ckpt(hparams.ckpt_path))
        model.eval()
        
        img = Image.open('data/coco/coco2017val/000000000285.jpg').convert('RGB')
        img_s = Image.open('data/wikiart/small/1.jpg').convert('RGB')
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(), # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # [-1, 1]
        ])
        img = tf(img).unsqueeze(0)
        img_s = tf(img_s).unsqueeze(0)
        # ic(img.shape)
        _, _, output = model(img, img_s)
        output = output[0].permute(1, 2, 0)
        output = (output + 1) / 2 # [-1, 1]->[0, 1]
        import matplotlib.pyplot as plt
        # plt.hist(output.reshape(-1).detach().cpu().numpy(), bins=255)
        plt.imshow(output.detach().cpu().numpy())
        plt.show()
        exit()
    
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
                      num_sanity_val_steps=0,
                      precision=32)

    trainer.fit(system, ckpt_path=hparams.ckpt_path) # if ckpt_path is not none, resume training.


