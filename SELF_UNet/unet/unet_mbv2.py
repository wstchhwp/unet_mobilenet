#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/9/5 0020 08:43
# @Author  : jiangdawei
# @FileName: mobilenetv2.py
# @Software: PyCharm

import torch
from torch import nn
import math

# todo 构造两次卷积+ BN + ReLu
# todo 左半边的下降　特征提取阶段
class CBR_2(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(CBR_2, self).__init__()

        self.conv_2 =nn.Sequential(
            nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=outchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        x = self.conv_2(x)
        return x



# 将输入值调整为最接近基数值整数倍的数值
def _make_divisible(ch, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_ch = max(min_value, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# 正常　卷积 + BN + 激活层
class CBR_Block(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        # 依据卷积核尺寸确定 padding
        padding = (kernel_size - 1) // 2
        super(CBR_Block, self).__init__(
            # kernel_size默认3，stride默认1，padding计算得到，groups默认等于1，bias不使用(因为下面有BN层)
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            # BN层输入特征矩阵深度为out_channel
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


# 倒残差结构块
class Inverted_Residual_Block(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):

        # expand_ratio为倍率因子，用来扩大深度
        super(Inverted_Residual_Block, self).__init__()

        # hidden_channel为第1层卷积层卷积核个数
        hidden_channel = in_channel * expand_ratio

        # 判断是否使用short cut分支
        # 步长为1 且 输入 输出通道数 相等,进行残茶边的叠加
        if stride == 1 and in_channel == out_channel:
            self.use_shortcut =True
        else:
            self.use_shortcut = False

        layers = []

        # 除了第一个堆叠的倒残差模块,expand_ratio为1 ,其他堆叠的倒残差模块expand_ratio都是6,都需要倒残差结构中第1层1×1卷积层
        if expand_ratio != 1:
            # 1x1 pointwise conv
            # 第1层：1×1卷积
            layers.append(CBR_Block(in_planes=in_channel, out_planes=hidden_channel, kernel_size=1, stride=1, groups=1))


        layers.extend([
            # extend与append功能相同，但extend能一次性批量插入很多元素
            # 3x3 depthwise 卷积。输入与输出通道数相同，都是hidden_channel, groups=hidden_channel 决定DW 卷积区别于普通卷积,当为1 时候为普通卷积
            CBR_Block(in_planes=hidden_channel, out_planes=hidden_channel, kernel_size=3, stride=stride, groups=hidden_channel),

            # 1x1 pointwise conv(linear),这里是线性激活函数，就不可以用刚才定义的CBR_Block()函数，这里用Conv2d。
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),

            nn.BatchNorm2d(out_channel),
            # 线性激活函数y=x，也就是不做处理。则不添加激活函数就相当于是线性激活函数。
        ])

        self.conv = nn.Sequential(*layers)


    def forward(self, x):
        # 判断是否满足short cut分支连接条件
        if self.use_shortcut:

            return x + self.conv(x)
        else:
            return self.conv(x)




class MobileNetV2(nn.Module):
    #   num_classes分类的类别个数
    #   alpha超参数,控制卷积层所使用卷积核个数的倍率,
    #   round_nearest为基数
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        self.n_channels = 3

        super(MobileNetV2, self).__init__()

        block = Inverted_Residual_Block

        # _make_divisible :将输入的卷积核个数调整为round_nearest的整数倍
        # input_channel表示表格中第1行Conv2d卷积层所使用的卷积核的个数，也等于下一层输入特征矩阵的深度
        input_channel = _make_divisible(32 * alpha, round_nearest)


        # last_channel表示表格中倒数第3行1×1卷积层的卷积核个数
        last_channel = _make_divisible(1280 * alpha, round_nearest)


        # 创建1个list列表，list列表中每一个元素就是表格中bottleneck对应每一行的参数t,c,n,s
        inverted_residual_setting = [
            # 扩展因子 输出通道　重复次数　步长
            # t, 　　　　　c, 　　　n, 　　　s

            #  当输入为512＊512 , size as follow
            # 256,256,32 变成 256,256,16
            [1, 16, 1, 1],         # 从开始算,到[1, 16, 1, 1],结束后,是总共有1 +1=2 层,从0算的话,是角标0-1,输出是256*256*16

            # 256,256,16 变成128,128,24
            [6, 24, 2, 2],        # 从开始算,到[6, 24, 2, 1],结束后,是总共有1 + (1+2)=4 层,从0算的话,是角标0-3,输出是128*128*24
            # 128,128,24变成 64,64,32
            [6, 32, 3, 2],        # 从开始算,到[6, 32, 3, 2],结束后,是总共有1 +(1+2+3) =7 层,从0算的话,是角标0-6,输出是64*64*32

            # 64,64,32 变成 32,32,64
            [6, 64, 4, 2],         # 从开始算,到[6, 64, 4, 2],结束后,是总共有1 +(1+2+3+4) =11 层,从0算的话,是角标0-10,输出是32*32*64

            # 32,32,64 变成 32,32,96
            [6, 96, 3, 1],         # 从开始算,到[6, 96, 3, 1],结束后,是总共有1 +(1 +2 +3 +4+ 3 ) =14 层,从0算的话,是角标0-13,输出是32*32*96

            # 32,32,96变成 16,16,160
            [6, 160, 3, 2],        # 从开始算,到[6, 160, 3, 2],结束后,是总共有1 +(1 +2 +3 +4+ 3 +3) =17 层,从0算的话,是角标0-16,输出是16*16*160

            # 16,16,160 变成 16,16,320

            [6, 320, 1, 1],        # 从开始算,到[6, 32, 3, 2],结束后,是总共有1 +(1 +2 +3 +4+ 3 +3 + 1)  =18 层,从0算的话,是角标0-17,输出是13*13*320
        ]

        #todo -----------------------------------特征提取层 开始

        # 列表 用于临时 存放特征提取层
        features = []

        # todo: 首先在fearures中添加第1个卷积层conv2d (普通卷积)
        # 卷积核尺寸为3，步长为2--》   输出尺寸减半，输出通道数目为input_channel，这里是32

        # 　todo 512,512,3 变成 256,256,32
        features.append(CBR_Block(in_planes = 3, out_planes=input_channel, kernel_size=3, stride=2, groups=1))

        # 接下来定义一系列b倒残差结构
        # t, 　　　　　c, 　　　n, 　　　s
        # 扩展因子 输出通道　重复次数　步长
        for expand_ratio_temp, out_channel_temp, repeat_num, stride_temp in inverted_residual_setting:

            # 将输出的channel个数通过_make_divisible进行调整
            output_channel = _make_divisible(out_channel_temp * alpha, round_nearest)

            # 循环搭建倒残差结构
            #依据repeat_num次数，判断 重复倒残差结构的次数
            for i in range(repeat_num):

                # stride 第1层倒残差结构的步距，其它层的步距都为1
                # 若存在堆叠残差块的情况，则只有在堆叠的第一次，根据s，设置步长，在同一个堆叠过程中，其它的倒残差块的步长都为1
                if i == 0:
                    stride = stride_temp
                else:
                    stride = 1


                # 在features中添加一系列的倒残差结构，block即上面定义的倒残差结构
                features.append(block(in_channel=input_channel, out_channel=output_channel, stride=stride, expand_ratio=expand_ratio_temp))

                # 每次经过一个倒残差后，都需要更新下下一次的输入通道，即为上一次的输出通道
                input_channel = output_channel


        # 构造完所有的堆叠倒残差时候，接下来需要进行1*1的卷积，输入通道为堆叠倒残差的最后输出通道，输出通道定义为last_channel 即1280
        # 保存宽高 尺寸不变 ,通道数input_channel 变为 last_channel
        # group=1 为普通卷积,非dw卷积

        # todo 13,13,320变成  13,13,1280

        features.append(CBR_Block(in_planes=input_channel, out_planes=last_channel, kernel_size=1, stride=1, groups=1))


        # 将所有的特征提取层全部合起来
        # todo features
        # 总共有 层
        # 1   (1 2 3 4 3 3 1 ) + 1

        self.features = nn.Sequential(*features)

        # todo -----------------------------------特征提取层 结束


        # 分类器：
        # 一个平均池化下采样 +  一个全连接层
        #自适应平均池化下采样，
        # 输出特征矩阵高和宽都为1  变为1*1*1280
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            # Dropout层
            nn.Dropout(0.2),
            #全连接层,
            # num_classes为预测的分类类别个数
            nn.Linear(last_channel, num_classes)
        )


        # 初始化权重
        for single_module in self.modules():
            # 如果是卷积层，对权重进行凯明初始化
            if isinstance(single_module, nn.Conv2d):
                nn.init.kaiming_normal_(single_module.weight, mode='fan_out')
                # 如果有bias，将偏置设置为0
                if single_module.bias is not None:
                    nn.init.zeros_(single_module.bias)

            # 如果子模块是BN层
            elif isinstance(single_module, nn.BatchNorm2d):
                # 将方差设置为1
                nn.init.ones_(single_module.weight)
                # 将均值设置为0
                nn.init.zeros_(single_module.bias)
            # 如果子模块是全连接层
            elif isinstance(single_module, nn.Linear):
                # normal为正态分布函数，将权重调整为均值为0，方差为0.01的正态分布
                nn.init.normal_(single_module.weight, 0, 0.01)
                # 将偏置设置为0
                nn.init.zeros_(single_module.bias)


    def forward(self, x):
        # 特征提取
        x = self.features(x)
        # 平均池化下采样
        x = self.avgpool(x)

        # flatten
        x = torch.flatten(x, 1)

        # 最后经过分类器
        x = self.classifier(x)
        return x



"""
增大图像尺寸　减小维度（通道数）
"""
class Feature_Up(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Feature_Up, self).__init__()


        # todo 上采样 改变宽高尺寸
        """
        size：据不同的输入制定输出大小；
        scale_factor：指定输出为输入的多少倍数；
        mode：可使用的上采样算法，有nearest，linear，bilinear，bicubic 和 trilinear。默认使用nearest；
        align_corners ：如果为 True，输入的角像素将与输出张量对齐，因此将保存下来这些像素的值。
        """

        self.upsample= nn.Upsample(size=None, scale_factor=2, mode='bilinear', align_corners=True)

        # todo 改变通道数
        self.upsample_conv = nn.Conv2d(in_channels=inchannel, out_channels=outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.upsample_bn = nn.BatchNorm2d(outchannel)
        self.upsample_relu = nn.ReLU(inplace=True)

        # todo 合并后的两次　卷积　+BN　＋　ReLu
        self.conv2 = CBR_2(inchannel=inchannel, outchannel=outchannel)


    # x1 是通道数更大，尺寸更小的特征图
    # x2 是通道数更小，尺寸更大的特征图
    def forward(self, x1, x2):

        # x1 进行上采样　改变宽高尺寸　然后再加一层卷积改变通道数
        x1 = self.upsample(x1)
        x1 = self.upsample_conv(x1)
        x1 = self.upsample_bn(x1)
        x1 = self.upsample_relu(x1)

        # 然后进行相加融合　再卷积
        #　按行往后排列
        out = torch.cat([x2, x1], dim=1)
        # 合并完后进行　两次的3*3 卷积
        out = self.conv2(out)

        return out



class Unet_MobileNetV2(nn.Module):
    def __init__(self,num_classes):
        super(Unet_MobileNetV2, self).__init__()
        bb_mobilenetv2 = MobileNetV2()

        # 新增两个模型属性　不然程序会报错，但是两个参数是冗余的，后面可以删除掉
        self.n_channels = 3
        self.n_classes = 4

        self.num_class =num_classes

        # 将输入
        # 一开始上来进行的是２次的３*3 卷积　，没有下采样操作
        # 3 * 512 * 512-->64 * 512 * 512
        # todo 合并后的两次　卷积　+BN　＋　ReLu
        self.top_conv = CBR_2(inchannel=3, outchannel=64)


        # 输出256\128\ 64\32 四个特征图，这里的通道数　应该不一致　需要额外处理

        # 输出是256 * 256 * 16
        self.layer_256 =bb_mobilenetv2.features[0:2]  # 第０　和第１　层

        # 输出是256 * 256 * 16，期望是256 * 256*128
        self.trans_layer_256 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=128,kernel_size=1,stride=1,padding=0,bias=False))


        # 输出是128*128*24
        self.layer_128 = bb_mobilenetv2.features[2:4]  # 第2　和第3层

        # 输出是128*128*24，期望是128*128*256
        self.trans_layer_128 = nn.Sequential(nn.Conv2d(in_channels=24,out_channels=256,kernel_size=1,stride=1,padding=0,bias=False))



        # 输出是64*64*32
        self.layer_64 = bb_mobilenetv2.features[4:7]  # 第4 5 6层

        # 输出是64*64*32，期望是64*64*512
        self.trans_layer_64 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False))


        # 输出是32*32*96
        self.layer_32 = bb_mobilenetv2.features[7:14]  # 第7 8 9 10 11 13 13 层

        # 输出是32*32*96，期望是32 * 32*1024
        self.trans_layer_32 = nn.Sequential(nn.Conv2d(in_channels=96, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=False))


        self.final_conv = nn.Conv2d(in_channels=64, out_channels=self.num_class, kernel_size=1, stride=1, padding=0,bias=False)


        # 右边特征融合
        # 第1次上采样 + 2次卷积　，输出512通道
        self.up1 = Feature_Up(1024, 512)

        # 第1次上采样 + 2次卷积　，输出256通道
        self.up2 = Feature_Up(512, 256)

        # 第1次上采样 + 2次卷积　，输出128通道
        self.up3 = Feature_Up(256, 128)

        # 第1次上采样 + 2次卷积　，输出64通道
        self.up4 = Feature_Up(128, 64)

        self._initialize_weights()

    def forward(self,x):

        out0= self.top_conv(x)
        # print("top_conv.shape",out0.shape)


        # 输出是256 * 256 * 16
        out = self.layer_256(x)
        # print("layer_256.shape",out.shape)


        # todo 256 * 256*128
        out1 = self.trans_layer_256(out)
        # print("trans_layer_256.shape",out1.shape)


        #输出是128 * 128 * 24
        out = self.layer_128(out)
        # print("layer_128.shape",out.shape)


        # todo 128*128*256
        out2 = self.trans_layer_128(out)

        # print("trans_layer_128.shape",out2.shape)


        # 输出是64*64*32
        out = self.layer_64(out)
        # print("layer_64.shape",out.shape)


        # todo 64*64*512
        out3 = self.trans_layer_64(out)
        # print("layer_64.shape",out.shape)


        # 输出是32*32*96
        out = self.layer_32(out)
        # todo 32 * 32*1024
        out4 = self.trans_layer_32(out)




        # todo 右边四次　　上采样　＋　cat融合　＋卷积　　进行特征融合
        # 通道　1024-> 512
        #torch.Size([10, 512, 64, 64])
        # out4 [n, 1024, 32, 32]－>[n,512,64,64]
        # out3 [n, 512, 64, 64]
        # 相加[n,1024,64,64]->[n,512,64,64]
        out5 = self.up1(x1=out4, x2=out3)

        # 通道　512-> 256
        #out6 torch.Size([n, 256, 128, 128])
        out6 = self.up2(x1=out5, x2=out2)

        # 通道　256-> 128
        # out7 torch.Size([n, 128, 256, 256])
        out7 = self.up3(x1=out6, x2=out1)

        # 通道　128-> 64
        # out8 torch.Size([n, 64, 512, 512])
        out8 = self.up4(x1=out7, x2=out0)

        # 通道　64-> num_class

        # torch.Size([10, 4, 512, 512])
        final_out = self.final_conv(out8)

        return final_out


    def _initialize_weights(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)

            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)

            elif isinstance(w, nn.Linear):
                nn.init.normal_(w.weight, 0, 0.01)
                nn.init.zeros_(w.bias)




if __name__ =="__main__":

    model = Unet_MobileNetV2(num_classes=4)
    # n c h w
    in_img = torch.randn(10, 3, 512, 512)
    # in_img = torch.randn(10, 3, 512, 1024)
    #
    # in_img = torch.nn.Parameter(in_img.cuda())

    out = model(in_img)

    print("out.shape = ", out.shape)
