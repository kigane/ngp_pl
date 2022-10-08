from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .nerfpp import NeRFPPDataset
from .rtmv import RTMVDataset
from .style_transfer_dataset import StylizedDataest


dataset_dict = {
    'nerf': NeRFDataset,
    'nsvf': NSVFDataset,
    'colmap': ColmapDataset,
    'nerfpp': NeRFPPDataset,
    'rtmv': RTMVDataset,
    'stylized': StylizedDataest       
}