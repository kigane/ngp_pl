import os
import torch
from torch.utils.data import Dataset
import numpy as np
import imageio


class NeRFRetDataset(Dataset):
    def __init__(self, ret_dir) -> None:
        super().__init__()
        self.rgb_img_paths = [f"{ret_dir}/{name}" for name in os.listdir(ret_dir) if name.endswith('png') and 'd' not in name]
    
    def __len__(self):
        return len(self.rgb_img_paths)
    
    def __getitem__(self, index):
        img = imageio.imread(self.rgb_img_paths[index]).astype(np.float32)/255.0
        return torch.FloatTensor(img)


class StylizedDataest(Dataset):
    #TODO
    def __init__(self, ret_dir) -> None:
        super().__init__()
        self.img_paths = [f"{ret_dir}/{name}" for name in os.listdir(ret_dir) if '_s' in name]
        self.poses = np.load(os.path.join(ret_dir, 'poses.npy'))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img = imageio.imread(self.img_paths[index]).astype(np.float32)/255.0
        img = torch.FloatTensor(img)
        sample = {
            'rgb': img,
            'pose': self.poses[index]
        }
        return sample


if __name__ == '__main__':
    ret_dir = 'results/nerf/lego'
    print([ret_dir+name for name in os.listdir(ret_dir) if name.endswith('png') and 'd' not in name])