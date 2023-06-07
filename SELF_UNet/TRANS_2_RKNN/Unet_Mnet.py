import torch.nn as nn
import math
import torch
import torch.nn.functional as F


# +

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.double_conv = DoubleConv(in_channels, 32)
        self.conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.up(x)
        x = self.double_conv(x)
        return self.conv(x)
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
        #nn.ReLU6(inplace=True)
    )


# +
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
        #nn.ReLU6(inplace=True)
    )
def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                #nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                #nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                #nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class Unet_Mnet_model(nn.Module):
    def __init__(self, n_classes=3, width_mult=1.):
        super(Unet_Mnet_model, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = True
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
#         assert input_size % 32 == 0
        # input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
#         self.features = [conv_bn(3, input_channel, 2)]
        self.in_conv = conv_bn(3, input_channel, 2)
    
        Mnet_block_list = []
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            block_temp_list = []
            for i in range(n):
                if i == 0:
#                     self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                    block_temp_list.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
#                     self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                    block_temp_list.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
            Mnet_block_list.append(nn.Sequential(*block_temp_list))

        
        self.down_conv_1 = (Mnet_block_list[0])
        self.down_conv_2 = (Mnet_block_list[1])
        self.down_conv_3 = (Mnet_block_list[2])
        self.down_conv_4 = (Mnet_block_list[3])
        self.down_conv_5 = (Mnet_block_list[4])
        self.down_conv_6 = (Mnet_block_list[5])
        self.down_conv_7 = (Mnet_block_list[6])
#         in 320 + 96 = 416 out (320 + 96)/2 = 208
        self.up_conv_1 = Up(416, 208, True)
#         in 208 + 32 = 240 out (208 + 32)/2 = 120
        self.up_conv_2 = Up(240, 120, True)
#         in 120 + 24 = 144 out (120 + 24)/2 = 72
        self.up_conv_3 = Up(144, 72, True)
#         in 72 + 16 = 88 out (72 + 16)/2 = 44
        self.up_conv_4 = Up(88, 44, True)
        self.out_conv = OutConv(44, n_classes)
#         import pdb
#         pdb.set_trace()
        self._initialize_weights()
        

    def forward(self, x):
        
        x_1 = self.in_conv(x)
        x_2 = self.down_conv_1(x_1)
        x_3 = self.down_conv_2(x_2)
        x_4 = self.down_conv_3(x_3)
        x_5 = self.down_conv_4(x_4)
        x_6 = self.down_conv_5(x_5)
        x_7 = self.down_conv_6(x_6)
        x_8 = self.down_conv_7(x_7)
#         print(x_8.cpu().shape)
#         print(x_6.cpu().shape)
        x_9 = self.up_conv_1(x_8, x_6)
        x_10 = self.up_conv_2(x_9, x_4)
        x_11 = self.up_conv_3(x_10, x_3)
        x_12 = self.up_conv_4(x_11, x_2)
        logtic = self.out_conv(x_12)
        
        return logtic

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
                
                
