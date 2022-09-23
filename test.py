from datetime import datetime
from icecream import ic
import models.transfer_net as tnet
import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # 关闭warning
warnings.filterwarnings("ignore", category=UserWarning) # 关闭warning

ic.configureOutput(prefix=lambda: datetime.now().strftime('%y-%m-%d %H:%M:%S | '),
       includeContext=False)

if __name__ == '__main__':
    # ic(tnet.vgg)
    # ic(tnet.vgg_layers_name_index_map)
    model = tnet.VGGEncoder("data/pretrained/vgg_normalised.pth")
    x = torch.randn((1, 3, 224, 224))
    out = model(x)
    for y in out:
        print(y.shape)