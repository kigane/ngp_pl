import os

import numpy as np
import torch
from tqdm import tqdm
import json
import glob
from PIL import Image

from .base import BaseDataset
from .ray_utils import *
from .color_utils import read_image
from .colmap_utils import read_cameras_binary


class StylizedDataest(BaseDataset):
    def __init__(self, root_dir, imgs_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        self.imgs_dir = imgs_dir
        self.dataset_name = kwargs.get('dataset_name', '')

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        if self.dataset_name == 'nerf':
            with open(os.path.join(self.root_dir, "transforms_train.json"), 'r') as f:
                meta = json.load(f)

            w = h = int(800*self.downsample)
            fx = fy = 0.5*800/np.tan(0.5*meta['camera_angle_x'])*self.downsample

            K = np.float32([[fx, 0, w/2],
                            [0, fy, h/2],
                            [0,  0,   1]])

            self.K = torch.FloatTensor(K)
            self.directions = get_ray_directions(h, w, self.K)
            self.img_wh = (w, h)
        elif self.dataset_name == 'nerfpp':
            K = np.loadtxt(glob.glob(os.path.join(self.root_dir, 'train/intrinsics/*.txt'))[0],
                       dtype=np.float32).reshape(4, 4)[:3, :3]
            K[:2] *= self.downsample
            w, h = Image.open(glob.glob(os.path.join(self.root_dir, 'train/rgb/*'))[0]).size
            w, h = int(w*self.downsample), int(h*self.downsample)
            self.K = torch.FloatTensor(K)
            self.directions = get_ray_directions(h, w, self.K)
            self.img_wh = (w, h)
        elif self.dataset_name == 'colmap':
            # Step 1: read and scale intrinsics (same for all images)
            camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
            h = int(camdata[1].height*self.downsample)
            w = int(camdata[1].width*self.downsample)
            self.img_wh = (w, h)

            if camdata[1].model == 'SIMPLE_RADIAL':
                fx = fy = camdata[1].params[0]*self.downsample
                cx = camdata[1].params[1]*self.downsample
                cy = camdata[1].params[2]*self.downsample
            elif camdata[1].model in ['PINHOLE', 'OPENCV']:
                fx = camdata[1].params[0]*self.downsample
                fy = camdata[1].params[1]*self.downsample
                cx = camdata[1].params[2]*self.downsample
                cy = camdata[1].params[3]*self.downsample
            else:
                raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
            self.K = torch.FloatTensor([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0,  0,  1]])
            self.directions = get_ray_directions(h, w, self.K)
        elif self.dataset_name == 'nsvf':
            if 'Synthetic' in self.root_dir or 'Ignatius' in self.root_dir:
                with open(os.path.join(self.root_dir, 'intrinsics.txt')) as f:
                    fx = fy = float(f.readline().split()[0]) * self.downsample
                if 'Synthetic' in self.root_dir:
                    w = h = int(800*self.downsample)
                else:
                    w, h = int(1920*self.downsample), int(1080*self.downsample)

                K = np.float32([[fx, 0, w/2],
                                [0, fy, h/2],
                                [0,  0,   1]])
            else:
                K = np.loadtxt(os.path.join(self.root_dir, 'intrinsics.txt'),
                            dtype=np.float32)[:3, :3]
                if 'BlendedMVS' in self.root_dir:
                    w, h = int(768*self.downsample), int(576*self.downsample)
                elif 'Tanks' in self.root_dir:
                    w, h = int(1920*self.downsample), int(1080*self.downsample)
                K[:2] *= self.downsample

            self.K = torch.FloatTensor(K)
            self.directions = get_ray_directions(h, w, self.K)
            self.img_wh = (w, h)
        else:
            raise NotImplementedError(f'Unsupported dataset_name {self.dataset_name}')

    def read_meta(self, split, **kwargs):
        if kwargs.get('use_guided_filter', False):
            sflag = '_f_'
        else:
            sflag = '_s_'
        img_paths = [os.path.join(self.imgs_dir, name) for name in sorted(os.listdir(os.path.join(self.imgs_dir))) if sflag in name]
        # 添加相应深度图
        poses = np.load(os.path.join(self.imgs_dir, 'poses.npy'))
        depths = np.load(os.path.join(os.path.dirname(self.imgs_dir), 'depths.npy'))
        
        self.rays = []
        self.depths = []
        # use every 8th image as test set
        # if split=='train':
        #     self.poses = poses
        #     # img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
        #     # self.poses = np.array([x for i, x in enumerate(poses) if i%8!=0])
        # elif split=='test': #! 迭代时所有的poses都渲染出来
        #     img_paths = img_paths
        #     self.poses = poses
        
        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            #! blend_a False => True, while apply mask
            img = read_image(img_path, self.img_wh)
            img = torch.FloatTensor(img)
            self.rays += [img]
        
        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.depths = depths # (N_images, hw) results['depth']结果就是这个形状
        self.poses = torch.FloatTensor(poses) # (N_images, 3, 4)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]
            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            rays = self.rays[img_idxs, pix_idxs]
            depths = self.depths[img_idxs, pix_idxs]
            sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs,
                      'depth': depths,
                      'rgb': rays[:, :3]}
            if self.rays.shape[-1] == 4: # HDR-NeRF data
                sample['exposure'] = rays[:, 3:]
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays)>0: # if ground truth available
                rays = self.rays[idx]
                sample['rgb'] = rays[:, :3]
                if rays.shape[1] == 4: # HDR-NeRF data
                    sample['exposure'] = rays[0, 3] # same exposure for all rays

        return sample
        

if __name__ == '__main__':
    ret_dir = 'results/nerf/lego'
    print([ret_dir+name for name in os.listdir(ret_dir) if name.endswith('png') and 'd' not in name])
