import torch
import torch.nn as nn

from .utils import adaptive_instance_normalization as adain
from .utils import calc_mean_std


vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2

    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2

    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4

    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4

    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


vgg_layers_names = [
    'conv0_1',
    'pad', 'conv1_1', 'relu1_1',
    'pad', 'conv1_2', 'relu1_2',
    'pool',
    'pad', 'conv2_1', 'relu2_1',
    'pad', 'conv2_2', 'relu2_2',
    'pool',
    'pad', 'conv3_1', 'relu3_1',
    'pad', 'conv3_2', 'relu3_2',
    'pad', 'conv3_3', 'relu3_3',
    'pad', 'conv3_4', 'relu3_4',
    'pool',
    'pad', 'conv4_1', 'relu4_1',
    'pad', 'conv4_2', 'relu4_2',
    'pad', 'conv4_3', 'relu4_3',
    'pad', 'conv4_4', 'relu4_4',
    'pool',
    'pad', 'conv5_1', 'relu5_1',
    'pad', 'conv5_2', 'relu5_2',
    'pad', 'conv5_3', 'relu5_3',
    'pad', 'conv5_4', 'relu5_4',
]

vgg_layers_name_index_map = {name:i+1 for i, name in enumerate(vgg_layers_names)}


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


class VGGEncoder(nn.Module):
    def __init__(self, 
        ckpt_path, # 预训练权重位置
        # 要使用的特征层
        features=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'conv4_2', 'relu5_1'] 
        ):
        super().__init__()
        self.features = features
        layer_index = [vgg_layers_name_index_map[name] for name in features]

        vgg.load_state_dict(torch.load(ckpt_path))
        vgg_pretrained = nn.Sequential(*list(vgg.children())[:layer_index[-1]])

        enc_layers = list(vgg_pretrained.children())
        # 构造enc_1, enc_2, ....，提取指定层的特征
        layer_index = [0] + layer_index
        for i in range(len(features)):
            setattr(self, features[i], nn.Sequential(*enc_layers[layer_index[i]:layer_index[i+1]]))
        
        # fix the encoder
        for name in features:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        x: input image
        return: feature maps of input 
        """
        results = [x]
        for i in range(len(self.features)):
            func = getattr(self, self.features[i])
            results.append(func(results[-1]))
        return results[1:]


class StyleTransferNet(nn.Module):
    """
    AdaIN Net
    """
    def __init__(self, encoder, decoder):
        super(StyleTransferNet, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        #! 用relu4_1的特征图计算损失 
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        #! 用relu1_1, relu2_1, relu3_1, relu4_1的均值和方差计算损失
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s
