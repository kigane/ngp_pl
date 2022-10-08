from datetime import datetime
import glob
import os
import shutil

import imageio
import numpy as np
import torch
# optimizer, losses
from apex.optimizers import FusedAdam
from einops import rearrange
# models
from kornia.utils.grid import create_meshgrid3d
import yaml
from gayts import neural_style_transfer
from adain_inference import style_transfer_one_image
from pama_inference import pama_infer_one_image
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
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
from losses import NeRFL1Loss, NeRFLoss
from models.networks import NGP
from models.rendering import MAX_SAMPLES, render
from utils import depth2img, load_ckpt, load_ngp_ckpt,slim_ckpt, parse_args, save_video, get_logger
from icecream import ic, install
from tqdm import tqdm

install()
ic.configureOutput(prefix=lambda: datetime.now().strftime('%y-%m-%d %H:%M:%S | '),
                   includeContext=True)

import warnings; warnings.filterwarnings("ignore")


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16
        if hparams.use_l1_loss:
            flogger.debug("use NeRFL1Loss")
            self.loss = NeRFL1Loss(lambda_distortion=self.hparams.distortion_loss_w)
        else:
            flogger.debug("use NeRFLoss(mse)")
            self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act)
        # 注册两个buffer-- density_grid & grid_coords
        G = self.model.grid_size # 128
        self.model.register_buffer('density_grid', # multi-scale：cascades层，每层grid的大小为G**3
            torch.zeros(self.model.cascades, G**3))
        # ic(self.model.density_grid)
        self.model.register_buffer('grid_coords', # (1, G, G, G, 3) -> (G**3, 3)
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    def forward(self, batch, split):
        """
        get volume rendering result
        """
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']] # ray的direction是由像素位置决定的
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]
        rays_o, rays_d = get_rays(directions, poses)
        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        """
        setup dataset for each machine
        """
        if hparams.loop < 1:
            dataset = dataset_dict[self.hparams.dataset_name]
            kwargs = {'root_dir': self.hparams.root_dir,
                    'downsample': self.hparams.downsample,
                    'st': hparams.loop > -1}
        else:
            dataset = dataset_dict['stylized']
            imgs_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/{self.hparams.loop-1}' # 上一轮渲染结果
            kwargs = {'root_dir': self.hparams.root_dir,
                    'imgs_dir': imgs_dir,
                    'dataset_name': self.hparams.dataset_name,
                    'downsample': self.hparams.downsample,
                    'st': hparams.loop > -1}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy

        self.test_dataset = dataset(split='test', **kwargs)
        #! 保存渲染图片的大小
        self.image_wh = self.train_dataset.img_wh
        # ic(self.image_wh)

    def configure_optimizers(self):
        """
        define optimizers and LR schedulers
        """
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        if hparams.loop > 0:
            load_ngp_ckpt(self.model, hparams.ckpt_path_pre)
        else:
            load_ckpt(self.model, self.hparams.weight_path)
        
        #! 固定density net
        if hparams.loop > 0 and hparams.fix_encoder:
            self.model.fix_xyz_encoder()

        net_params = []
        for n, p in self.named_parameters():
            print(f"{n}:{p.shape}")
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    self.hparams.lr/30)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        if self.hparams.loop < 1 or not self.hparams.fix_encoder:
            #! 这个方法会重置density_grid, 要想训练一次后不再更新，则之后不用调用该方法
            self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)

    def training_step(self, batch, batch_idx, *args):
        """
        the complete training loop
        batch: dataloader的内容
        batch_idx: 索引
        """
        #! 希望desity_grid在用原始图片训练一次后不再更新，以保留原始的3D空间信息
        if  (not self.hparams.fix_encoder or\
            self.hparams.loop < 1) and\
            self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=self.hparams.dataset_name=='colmap')
        results = self(batch, split='train')
        loss_d = self.loss(results, batch, 
                           use_depth_loss=self.hparams.use_depth_loss)
        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())
        loss_dict = {k:lo.detach().cpu().mean() for k, lo in loss_d.items()}
        flogger.debug(f"loss depth: {loss_dict.get('depth', 0)}")
        
        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            # 区分NeRF和NeRF Style Transfer任务
            if self.hparams.loop == -1:
                self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/raw'
            else:
                self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/{self.hparams.loop}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        rgb_gt = batch['rgb']
        pose = batch['pose']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                           torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)
            if self.hparams.loop > -1:
                logs['pose'] = pose.cpu().numpy()
            # 保留原NeRF的深度图作为GT
            if self.hparams.loop == 0:
                logs['depth'] = results['depth'].cpu().numpy()
                
        return logs

    def validation_epoch_end(self, outputs):
        """
        outputs: validation_step返回值组成的list
        """
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)
        
        if self.hparams.loop > -1:
            img_pose = np.stack([x['pose'] for x in outputs], axis=0)
            np.save(f'{self.val_dir}/poses.npy', img_pose)
        
        if self.hparams.loop == 0:
            img_depths = np.stack([x['depth'] for x in outputs], axis=0)
            np.save(f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}/depths.npy', img_depths)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = parse_args()
    flogger = get_logger()

    # 开始新的NeRF Style Transfer时，删除旧文件
    if hparams.loop == 0:
        ckpt_dir = f'ckpts/{hparams.dataset_name}/{hparams.exp_name}'
        result_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}'
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)

    # add ckpt_path_pre, used in SNeRF training
    version = '' if hparams.loop < 2 else f'-v{hparams.loop-1}'
    hparams.ckpt_path_pre = f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}{version}.ckpt'
    
    print('\033[32m#################################################\033[0m')
    print(f'\033[32m##################LOOP {hparams.loop}#########################\033[0m')
    print('\033[32m#################################################\033[0m')
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]
    
    if hparams.use_wandb:
        logger = WandbLogger(save_dir=f"logs/{hparams.dataset_name}",
                             project="NeRF",
                             name=hparams.exp_name,
                             log_model=True)  # 最后保存一次模型
    else:
        logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
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
                      precision=16)

    trainer.fit(system, ckpt_path=hparams.ckpt_path) # if ckpt_path is not none, resume training.
    # ic(system.model.density_grid)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)
    
    # 不进行StyleTransfer
    if hparams.loop == -1:
        exit()
    
    result_dir = system.val_dir
    # 渲染图路径
    rgb_paths = [f"{result_dir}/{name}" for name in os.listdir(result_dir) if name.endswith('png') and 'd' not in name]
    
    # Stylized Image输出到相同目录下
    hparams.out_dir = result_dir
    hparams.image_wh = system.image_wh
    flogger.info(f"out_dir={hparams.out_dir}")
    flogger.info(f"image_wh={hparams.image_wh}")
    # 保存超参数
    with open(os.path.join(os.path.dirname(result_dir), 'config.yml'), 'w') as f:
        yaml.safe_dump(hparams.__dict__, f)
    
    print("开始风格迁移")
    
    if hparams.style_transfer_method == 'gayts':
        style_transfer = neural_style_transfer
    elif hparams.style_transfer_method == 'adain':
        style_transfer = style_transfer_one_image
    elif hparams.style_transfer_method == 'pama':
        style_transfer = pama_infer_one_image
    else:
        raise NotImplementedError(f"{hparams.style_transfer_method} is not supported")
    
    for path in tqdm(rgb_paths):
        hparams.content_image = path #? 我为什么要写这一行
        style_transfer(path, hparams.style_image, hparams)
    
    # 保存一下NeRF渲染结果和单独风格化后结果的视频
    save_video(hparams)