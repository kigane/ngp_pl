from datetime import datetime
import glob
import os
import warnings

import imageio
import numpy as np
import torch
from tqdm import tqdm
# optimizer, losses
from apex.optimizers import FusedAdam
from einops import rearrange
from icecream import ic
# models
from kornia.utils.grid import create_meshgrid3d
from torch.optim.lr_scheduler import CosineAnnealingLR
# data
from torch.utils.data import DataLoader
# metrics
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from datasets import dataset_dict
from datasets.ray_utils import get_rays
from losses import NeRFLoss
from models.networks import NGP
from models.rendering import MAX_SAMPLES, render
from utils import depth2img, load_ckpt, parse_args, slim_ckpt
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def forward_render(batch, split, buffers):
    if split=='train':
        poses = buffers['poses'][batch['img_idxs']]
        # ray的direction是由像素位置决定的
        directions = buffers['directions'][batch['pix_idxs']] 
    else:
        poses = batch['pose']
        directions = buffers['directions']

    rays_o, rays_d = get_rays(directions, poses)

    kwargs = {'test_time': split!='train',
                'random_bg': hparams.random_bg}
    if hparams.scale > 0.5:
        kwargs['exp_step_factor'] = 1/256
    if hparams.use_exposure:
        kwargs['exposure'] = batch['exposure']

    return render(model, rays_o, rays_d, **kwargs)

if __name__ == '__main__':
    hparams = parse_args()
    ic.configureOutput(
        prefix=lambda: f"{datetime.now().strftime('%y-%m-%d %H:%M:%S')} | ",
        includeContext=False
    )
    ic(hparams)
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    
    logger = SummaryWriter(f"logs/{hparams.dataset_name}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in range(hparams.loop_times):
        #---------------------train nerf-------------------------------------
        print("---------------------------------------------")
        print("------------nerf training start--------------")
        print("---------------------------------------------")
        #--------set up--------
        warmup_steps = 256
        update_interval = 16

        nerf_loss = NeRFLoss(lambda_distortion=hparams.distortion_loss_w)
        
        train_psnr = PeakSignalNoiseRatio(data_range=1).cuda()
        val_psnr = PeakSignalNoiseRatio(data_range=1).cuda()
        val_ssim = StructuralSimilarityIndexMeasure(data_range=1).cuda()
        if hparams.eval_lpips:
            val_lpips = LearnedPerceptualImagePatchSimilarity('vgg').cuda()
            for p in val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if hparams.use_exposure else 'Sigmoid'
        model = NGP(scale=hparams.scale, rgb_act=rgb_act)
        G = model.grid_size # 128
        model.register_buffer(
            'density_grid', # multi-scale：cascades层，每层grid的大小为G**3
            torch.zeros(model.cascades, G**3).to(device)
        )
        model.register_buffer(
            'grid_coords', # (1, G, G, G, 3) -> (G**3, 3)
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3).to(device)
        )
        #--------datasets---------
        dataset = dataset_dict[hparams.dataset_name]
        kwargs = {'root_dir': hparams.root_dir,
                  'downsample': hparams.downsample}
        
        train_dataset = dataset(split=hparams.split, **kwargs)
        train_dataset.batch_size = hparams.batch_size
        train_dataset.ray_sampling_strategy = hparams.ray_sampling_strategy
        train_loader = DataLoader(train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)
        
        test_dataset = dataset(split='test', **kwargs)
        test_loader =  DataLoader(test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)
        #--------optimizer---------
        buffers = {}
        buffers['directions'] = train_dataset.directions.to(device)
        buffers['poses'] = train_dataset.poses.to(device)

        load_ckpt(model, hparams.weight_path)

        net_opt = FusedAdam(model.parameters(), hparams.lr, eps=1e-15)
        net_sch = CosineAnnealingLR(net_opt,
                                    hparams.num_epochs,
                                    hparams.lr/30)
        #--------training----------
        model.mark_invisible_cells(train_dataset.K.to(device),
                                buffers['poses'],
                                train_dataset.img_wh)
        
        global_step = -1
        for epoch in range(hparams.num_epochs):
            pbar = tqdm(enumerate(train_loader))
            for step, batch in pbar:
                #--------train step---------
                global_step += 1
                if global_step%update_interval == 0:
                    ic(MAX_SAMPLES)
                    th = 0.01*MAX_SAMPLES/3**0.5
                    ic(th)
                    model.update_density_grid(th,
                                           warmup=global_step<warmup_steps,
                                           erode=hparams.dataset_name=='colmap')

                results = forward_render(batch, 'train', buffers)
                
                loss_d = nerf_loss(results, batch)
                
                loss = sum(lo.mean() for lo in loss_d.values())
                
                with torch.no_grad():
                    train_psnr(results['rgb'], batch['rgb'].cuda())
                logger.add_scalar('lr', net_opt.param_groups[0]['lr'])
                logger.add_scalar('train/loss', loss)
                # ray marching samples per ray (occupied space on the ray)
                logger.add_scalar('train/rm_s', results['rm_samples']/len(batch['rgb']))
                # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
                logger.add_scalar('train/vr_s', results['vr_samples']/len(batch['rgb']))
                logger.add_scalar('train/psnr', train_psnr.compute())
                train_psnr.reset()
        
        #--------validation---------
        torch.cuda.empty_cache()
        if not hparams.no_save_test:
            val_dir = f'results/{hparams.dataset_name}/{hparams.exp_name}'
            os.makedirs(val_dir, exist_ok=True)
        
        for batch in tqdm(test_loader):
            rgb_gt = batch['rgb']
            results = forward_render(batch, 'test', buffers)

            logs = {}
            # compute each metric per image
            val_psnr(results['rgb'], rgb_gt)
            logs['psnr'] = val_psnr.compute()
            val_psnr.reset()

            w, h = train_dataset.img_wh
            rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
            rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
            val_ssim(rgb_pred, rgb_gt)
            logs['ssim'] = val_ssim.compute()
            val_ssim.reset()
            if hparams.eval_lpips:
                val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                            torch.clip(rgb_gt*2-1, -1, 1))
                logs['lpips'] = val_lpips.compute()
                val_lpips.reset()

            if not hparams.no_save_test: # save test image to disk
                idx = batch['img_idxs']
                rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
                rgb_pred = (rgb_pred*255).astype(np.uint8)
                depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
                imageio.imsave(os.path.join(val_dir, f'{idx:03d}.png'), rgb_pred)
                imageio.imsave(os.path.join(val_dir, f'{idx:03d}_d.png'), depth)       
        
        psnrs = torch.stack([x['psnr'] for x in logs])
        mean_psnr = psnrs.mean()
        logger.add_scalar('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in logs])
        mean_ssim = ssims.mean()
        logger.add_scalar('test/ssim', mean_ssim)

        if hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in logs])
            mean_lpips = lpipss.mean()
            logger.add_scalar('test/lpips_vgg', mean_lpips)

        if not hparams.val_only: # save slimmed ckpt for the last epoch
            ckpt_ = \
                slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                        save_poses=hparams.optimize_ext)
            torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

        if (not hparams.no_save_test) and \
            hparams.dataset_name=='nsvf' and \
            'Synthetic' in hparams.root_dir: # save video
            imgs = sorted(glob.glob(os.path.join(val_dir, '*.png')))
            imageio.mimsave(os.path.join(val_dir, 'rgb.mp4'),
                            [imageio.imread(img) for img in imgs[::2]],
                            fps=30, macro_block_size=1)
            imageio.mimsave(os.path.join(val_dir, 'depth.mp4'),
                            [imageio.imread(img) for img in imgs[1::2]],
                            fps=30, macro_block_size=1)
        print("---------------------------------------------")
        print("------------nerf training end----------------")
        print("---------------------------------------------")
        #----------------------style transfer--------------------------------
        print("---------------------------------------------")
        print("------------style transfer start-------------")
        print("---------------------------------------------")
        
        print("---------------------------------------------")
        print("------------style transfer end---------------")
        print("---------------------------------------------")
        break
        
        