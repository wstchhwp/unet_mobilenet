import torch.nn as nn
import torch
import torch.functional as F

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



"""
降低图像尺寸　增大维度(通道数)
"""
class Feature_Down(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Feature_Down, self).__init__()

        """
        kernel_size 可以看做是一个滑动窗口，，如果输入是单个值，例如 3 ，那么窗口的大小就是 3 × 3 还可以输入元组，例如 (3, 2) ，那么窗口大小就是 3 × 2 ,最大池化的方法就是取这个窗口覆盖元素中的最大值。
        stride 如果不指定这个参数，那么默认步长跟最大池化窗口大小一致。如果指定了参数，那么将按照我们指定的参数进行滑动
        padding 控制如何进行填充，填充值默认为0。如果是单个值，例如 1，那么将在周围填充一圈0,还可以用元组指定如何填充
        """
        self.pool_conv = nn.Sequential(
            # 池化
            nn.MaxPool2d(kernel_size=2),
            CBR_2(inchannel=inchannel, outchannel=outchannel),
        )

    def forward(self,x):
        x = self.pool_conv(x)
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


class My_Unet(nn.Module):
    def __init__(self, num_class):
        super(My_Unet, self).__init__()

        # 新增两个模型属性　不然程序会报错，但是两个参数是冗余的，后面可以删除掉
        self.n_channels = 3
        self.n_classes = 4

        # 这个是自己设置的两个属性
        self.ori_channel = 3
        self.num_class = num_class


        # 输出通道数
        # 3 * 512 * 512
        # 一开始上来进行的是２次的３*3 卷积　，没有下采样操作
        # 3 * 512 * 512-->64 * 512 * 512
        self.top_conv = CBR_2(3, 64)

        # 64 * 512 * 512-->128 * 256 * 256
        # 第1次下采样 + 2次卷积　，输出128通道
        self.down1 = Feature_Down(64, 128)

        # 128 * 256 * 256-->256 * 128 * 128
        # 第2次下采样 + 2次卷积　，输出256通道
        self.down2 = Feature_Down(128, 256)

        # 256 * 128 * 128--> 512 * 64 * 64
        # 第3次下采样 + 2次卷积　，输出512通道
        self.down3 = Feature_Down(256, 512)


        # 512 * 64 * 64--> 1024 * 32 * 32
        # 第4次下采样 + 2次卷积　，输出1024通道
        self.down4 = Feature_Down(512, 1024)



        # 右边特征融合
        # 第1次上采样 + 2次卷积　，输出512通道
        self.up1 = Feature_Up(1024, 512)

        # 第1次上采样 + 2次卷积　，输出256通道
        self.up2 = Feature_Up(512, 256)

        # 第1次上采样 + 2次卷积　，输出128通道
        self.up3 = Feature_Up(256, 128)

        # 第1次上采样 + 2次卷积　，输出64通道
        self.up4 = Feature_Up(128, 64)


        self.final_conv = nn.Conv2d(in_channels=64, out_channels=self.num_class, kernel_size=1, stride=1, padding=0,bias=False)

    def forward(self, x):

        # 通道　3-> 64
        # 3 * 512 * 512-->64 * 512 * 512
        out0 = self.top_conv(x)   # torch.Size([n, 64, 512, 512])

        # todo 左边四次　　下采样＋卷积　　进行特征提取
        # 通道　64-> 128　
        # 64 * 512 * 512-->128 * 256 * 256
        out1 = self.down1(out0)   # torch.Size([n, 128, 256, 256])
        # 通道　128-> 256
        # 128 * 256 * 256-->256 * 128 * 128
        out2 = self.down2(out1)  # torch.Size([n, 256, 128, 128])

        # 通道　256-> 512　　　
        # 256 * 128 * 128--> 512 * 64 * 64　　　
        out3 = self.down3(out2)  # torch.Size([n, 512, 64, 64])

        # 通道　512-> 1024　　
        # 512 * 64 * 64--> 1024 * 32 * 32　
        out4 = self.down4(out3)  #torch.Size([10, 1024, 32, 32])



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
        out = self.final_conv(out8)
        return out




if __name__ == "__main__":

    # n c h w
    input = torch.randn(10, 3, 512, 512)

    model = My_Unet(num_class=4)
    out = model(input)
    print("out.size() = ",out.size())





















