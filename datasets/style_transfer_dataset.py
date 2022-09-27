import os

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json

from .base import BaseDataset
from .ray_utils import *
from .color_utils import read_image
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary


class SimpleDataset(Dataset):
    def __init__(self, dir) -> None:
        super().__init__()
        self.img_paths = os.listdir(dir)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img = imageio.imread(self.img_paths[index]).astype(np.float32)/255.0
        return torch.FloatTensor(img)


class NeRFRetDataset(Dataset):
    def __init__(self, ret_dir) -> None:
        super().__init__()
        self.rgb_img_paths = [f"{ret_dir}/{name}" for name in os.listdir(ret_dir) if name.endswith('png') and 'd' not in name]
    
    def __len__(self):
        return len(self.rgb_img_paths)
    
    def __getitem__(self, index):
        img = imageio.imread(self.rgb_img_paths[index]).astype(np.float32)/255.0
        return torch.FloatTensor(img)


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
        else:
            raise NotImplementedError(f'Unsupported dataset_name {self.dataset_name}')

    def read_meta(self, split, **kwargs):
        img_paths = [os.path.join(self.imgs_dir, name) for name in sorted(os.listdir(os.path.join(self.imgs_dir))) if '_s_' in name]
        poses = np.load(os.path.join(self.imgs_dir, 'poses.npy'))
        
        self.rays = []
        # use every 8th image as test set
        if split=='train':
            img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
            self.poses = np.array([x for i, x in enumerate(poses) if i%8!=0])
        elif split=='test': #! 迭代时所有的poses都渲染出来
            img_paths = img_paths
            self.poses = poses
        
        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            img = read_image(img_path, self.img_wh, blend_a=False)
            img = torch.FloatTensor(img)
            self.rays += [img]
            
        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)

        

if __name__ == '__main__':
    ret_dir = 'results/nerf/lego'
    print([ret_dir+name for name in os.listdir(ret_dir) if name.endswith('png') and 'd' not in name])
