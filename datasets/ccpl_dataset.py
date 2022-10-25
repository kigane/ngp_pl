import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def is_rgb_file(filename):
    return filename.endswith('.png') and '_d.' not in filename and '_s_' not in filename

def default_loader(path):
    return Image.open(path).convert('RGB')

class CCPLDataset(data.Dataset):
    def __init__(self,dataPath,loadSize,fineSize,test=False,video=False):
        super(CCPLDataset,self).__init__()
        self.dataPath = dataPath
        #! 为NeRF风格迁移，暂时先改成这样 
        if(video):
            self.image_list = [x for x in sorted(os.listdir(dataPath)) if is_rgb_file(x)]
        else:
            self.image_list = [x for x in os.listdir(dataPath) if is_image_file(x)]
            
        if not test:
            self.transform = transforms.Compose([
            		         transforms.Resize(fineSize),
            		         transforms.RandomCrop(fineSize),
                             transforms.RandomHorizontalFlip(),
            		         transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([
            		         transforms.Resize(fineSize),
            		         transforms.ToTensor()])

        self.test = test

    def __getitem__(self,index): 
        dataPath = os.path.join(self.dataPath,self.image_list[index])

        Img = default_loader(dataPath)
        ImgA = self.transform(Img)

        imgName = self.image_list[index]
        imgName = imgName.split('.')[0] # 'frame_0001'
        return ImgA,imgName

    def __len__(self):
        return len(self.image_list)
