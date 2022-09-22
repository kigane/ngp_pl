from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .nerfpp import NeRFPPDataset
from .rtmv import RTMVDataset
from .style_transfer import NeRFRetDataset, StylizedDataest


dataset_dict = {
    'nerf': NeRFDataset,
    'nsvf': NSVFDataset,
    'colmap': ColmapDataset,
    'nerfpp': NeRFPPDataset,
    'rtmv': RTMVDataset,
    'nerf_ret': NeRFRetDataset,
    'stylied': StylizedDataest       
}