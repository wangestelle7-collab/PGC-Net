import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import time


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.up(x)


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU()
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UPAttention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, x_c, k):
        super(UPAttention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.change = nn.ConvTranspose2d(x_c, F_l, kernel_size=k, stride=k, padding=0, bias=False)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, g, x):
        x = self.change(x)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Generator_com_diff(nn.Module):
    def __init__(self, img_ch=3, output_ch=3, t=2):
        super(Generator_com_diff, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down11 = RRCNN_block(ch_in=img_ch, ch_out=32, t=t)
        self.down21 = RRCNN_block(ch_in=32, ch_out=64, t=t)
        self.down31 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.down41 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.down51 = RRCNN_block(ch_in=256, ch_out=512, t=t)

        self.Up51 = up_conv(ch_in=512, ch_out=256)
        self.att41 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.updown51 = RRCNN_block(ch_in=512, ch_out=256, t=t)

        self.Up41 = up_conv(ch_in=256, ch_out=128)
        self.att31 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Att471 = UPAttention_block(F_g=128, F_l=128, F_int=64, x_c=256, k=2)
        self.updown61 = RRCNN_block(ch_in=384, ch_out=128, t=t)

        self.Up31 = up_conv(ch_in=128, ch_out=64)
        self.att21 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Att381 = UPAttention_block(F_g=64, F_l=64, F_int=32, x_c=128, k=2)
        self.Att481 = UPAttention_block(F_g=64, F_l=64, F_int=32, x_c=256, k=4)
        self.updown71 = RRCNN_block(ch_in=256, ch_out=64, t=t)

        self.Up21 = up_conv(ch_in=64, ch_out=32)
        self.att11 = Attention_block(F_g=32, F_l=32, F_int=16)
        self.Att291 = UPAttention_block(F_g=32, F_l=32, F_int=16, x_c=64, k=2)
        self.Att391 = UPAttention_block(F_g=32, F_l=32, F_int=16, x_c=128, k=4)
        self.Att491 = UPAttention_block(F_g=32, F_l=32, F_int=16, x_c=256, k=8)
        self.updown81 = RRCNN_block(ch_in=160, ch_out=32, t=t)

        self.Conv_1x1_1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.down11(x)  # [1,32,256,256]
        x2 = self.down21(self.Maxpool(x1))  # [1,64,128,128]
        x3 = self.down31(self.Maxpool(x2))  # [1,128,64,64]
        x4 = self.down41(self.Maxpool(x3))  # [1,256,32,32]
        x5 = self.down51(self.Maxpool(x4))  # [1,512,16,16]

        x6 = self.Up51(x5)  # [1,256,32,32]
        x4_6 = self.att41(g=x6, x=x4)  # [1,256,32,32]
        x6 = torch.cat([x4_6, x6], dim=1)  # [1,512,32,32]
        x6 = self.updown51(x6)  # [1,256,32,32]

        x7 = self.Up41(x6)  # [1,128,64,64]
        x3_7 = self.att31(g=x7, x=x3)  # [1,128,64,64]
        x4_7 = self.Att471(g=x7, x=x4)  # [1,128,64,64]
        x7 = torch.cat([x7, x3_7, x4_7], dim=1)  # [1,384,64,64]
        x7 = self.updown61(x7)  # [1,128,64,64]

        x8 = self.Up31(x7)  # [1,64,128,128]
        x2_8 = self.att21(g=x8, x=x2)  # [1,64,128,128]
        x3_8 = self.Att381(g=x8, x=x3)  # [1,64,128,128]
        x4_8 = self.Att481(g=x8, x=x4)  # [1,64,128,128]
        x8 = torch.cat([x8, x2_8, x3_8, x4_8], dim=1)  # [1,256,128,128]
        x8 = self.updown71(x8)  # [1,64,128,128]

        x9 = self.Up21(x8)  # [1,32,256,256]
        x1_9 = self.att11(g=x9, x=x1)  # [1,32,256,256]
        x2_9 = self.Att291(g=x9, x=x2)  # [1,32,256,256]
        x3_9 = self.Att391(g=x9, x=x3)  # [1,32,256,256]
        x4_9 = self.Att491(g=x9, x=x4)  # [1,32,256,256]
        x9 = torch.cat([x9, x1_9, x2_9, x3_9, x4_9], dim=1)  # [1,160,256,256]
        x9 = self.updown81(x9)  # [1,32,256,256]

        mid = self.Conv_1x1_1(x9)
        merge = torch.cat([mid, x], dim=1)
        return x5, mid, merge


class UnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        ker_s_1 = 16
        pad_1 = int(ker_s_1 / 2 - 1)
        ker_s_2 = 12
        pad_2 = int(ker_s_2 / 2 - 1)
        ker_s_3 = 8
        pad_3 = int(ker_s_3 / 2 - 1)
        ker_s_4 = 4
        pad_4 = int(ker_s_4 / 2 - 1)
        st = 2
        c_channal = 128

        self.wc1 = nn.Parameter(torch.ones([4]))
        self.wc2 = nn.Parameter(torch.ones([4]))
        self.wc3 = nn.Parameter(torch.ones([4]))
        self.wc4 = nn.Parameter(torch.ones([4]))

        self.cov11 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(input_nc, 16, ker_s_1, st, pad_1, bias=False),
            nn.BatchNorm2d(16)
        )
        self.cov12 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(input_nc, 16, ker_s_2, st, pad_2, bias=False),
            nn.BatchNorm2d(16)
        )
        self.cov13 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(input_nc, 16, ker_s_3, st, pad_3, bias=False),
            nn.BatchNorm2d(16)
        )
        self.cov14 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(input_nc, 16, ker_s_4, st, pad_4, bias=False),
            nn.BatchNorm2d(16)
        )
        self.se1 = SELayer(64)

        self.cov21 = nn.Sequential(
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, ker_s_1, st, pad_1, bias=False),
            nn.BatchNorm2d(32)
        )
        self.cov22 = nn.Sequential(
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, ker_s_2, st, pad_2, bias=False),
            nn.BatchNorm2d(32)
        )
        self.cov23 = nn.Sequential(
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, ker_s_3, st, pad_3, bias=False),
            nn.BatchNorm2d(32)
        )
        self.cov24 = nn.Sequential(
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, ker_s_4, st, pad_4, bias=False),
            nn.BatchNorm2d(32)
        )
        self.se2 = SELayer(128)

        self.cov31 = nn.Sequential(
            nn.Dropout(0.4),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, ker_s_1, st, pad_1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.cov32 = nn.Sequential(
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, ker_s_2, st, pad_2, bias=False),
            nn.BatchNorm2d(64)
        )
        self.cov33 = nn.Sequential(
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, ker_s_3, st, pad_3, bias=False),
            nn.BatchNorm2d(64)
        )
        self.cov34 = nn.Sequential(
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, ker_s_4, st, pad_4, bias=False),
            nn.BatchNorm2d(64)
        )
        self.se3 = SELayer(256)

        self.cov41 = nn.Sequential(
            nn.Dropout(0.4),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, ker_s_1, st, pad_1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.cov42 = nn.Sequential(
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, ker_s_2, st, pad_2, bias=False),
            nn.BatchNorm2d(128)
        )
        self.cov43 = nn.Sequential(
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, ker_s_3, st, pad_3, bias=False),
            nn.BatchNorm2d(128)
        )
        self.cov44 = nn.Sequential(
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, ker_s_4, st, pad_4, bias=False),
            nn.BatchNorm2d(128)
        )
        self.se4 = SELayer(512)

        ker_s_dc = 4
        pad_dc = int(ker_s_dc / 2 - 1)

        self.cen = nn.Sequential(
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Conv2d(512, 1024, ker_s_dc, st, pad_dc, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, ker_s_dc, st, pad_dc, bias=False),
            nn.BatchNorm2d(512),
            nn.Dropout(0.2)
        )

        self.dov4 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 256, ker_s_dc, st, pad_dc, bias=False),
            nn.BatchNorm2d(256),
            nn.Dropout(0.2)
        )

        self.dov3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, ker_s_3, st, pad_3, bias=False),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2)
        )
        self.dov2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, ker_s_2, st, pad_2, bias=False),
            nn.BatchNorm2d(64),
        )
        self.dov1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, output_nc, ker_s_1, st, pad_1, bias=False),
            # nn.Tanh()
        )

        self.end = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(6, output_nc, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        wc11 = torch.exp(self.wc1[0]) / torch.sum(torch.exp(self.wc1))
        wc12 = torch.exp(self.wc1[1]) / torch.sum(torch.exp(self.wc1))
        wc13 = torch.exp(self.wc1[2]) / torch.sum(torch.exp(self.wc1))
        wc14 = torch.exp(self.wc1[3]) / torch.sum(torch.exp(self.wc1))

        wc21 = torch.exp(self.wc2[0]) / torch.sum(torch.exp(self.wc2))
        wc22 = torch.exp(self.wc2[1]) / torch.sum(torch.exp(self.wc2))
        wc23 = torch.exp(self.wc2[2]) / torch.sum(torch.exp(self.wc2))
        wc24 = torch.exp(self.wc2[3]) / torch.sum(torch.exp(self.wc2))

        wc31 = torch.exp(self.wc3[0]) / torch.sum(torch.exp(self.wc3))
        wc32 = torch.exp(self.wc3[1]) / torch.sum(torch.exp(self.wc3))
        wc33 = torch.exp(self.wc3[2]) / torch.sum(torch.exp(self.wc3))
        wc34 = torch.exp(self.wc3[3]) / torch.sum(torch.exp(self.wc3))

        wc41 = torch.exp(self.wc4[0]) / torch.sum(torch.exp(self.wc4))
        wc42 = torch.exp(self.wc4[1]) / torch.sum(torch.exp(self.wc4))
        wc43 = torch.exp(self.wc4[2]) / torch.sum(torch.exp(self.wc4))
        wc44 = torch.exp(self.wc4[3]) / torch.sum(torch.exp(self.wc4))

        cx1 = self.se1(torch.cat([torch.cat([wc11 * self.cov11(x), wc12 * self.cov12(x)], 1),
                                  torch.cat([wc13 * self.cov13(x), wc14 * self.cov14(x)], 1)], 1))
        cx2 = self.se2(torch.cat([torch.cat([wc21 * self.cov21(cx1), wc22 * self.cov22(cx1)], 1),
                                  torch.cat([wc23 * self.cov23(cx1), wc24 * self.cov24(cx1)], 1)], 1))
        cx3 = self.se3(torch.cat([torch.cat([wc31 * self.cov31(cx2), wc32 * self.cov32(cx2)], 1),
                                  torch.cat([wc33 * self.cov33(cx2), wc34 * self.cov34(cx2)], 1)], 1))
        cx4 = self.se4(torch.cat([torch.cat([wc41 * self.cov41(cx3), wc42 * self.cov42(cx3)], 1),
                                  torch.cat([wc43 * self.cov43(cx3), wc44 * self.cov44(cx3)], 1)], 1))

        ce = self.cen(cx4)

        d4 = torch.cat([ce, cx4], 1)
        dx4 = self.dov4(d4)

        d3 = torch.cat([dx4, cx3], 1)
        dx3 = self.dov3(d3)

        d2 = torch.cat([dx3, cx2], 1)
        dx2 = self.dov2(d2)

        d1 = torch.cat([dx2, cx1], 1)
        dx1 = self.dov1(d1)

        return dx1


class Seg(nn.Module):
    def __init__(self, input_nc, output_nc, base_channels=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(Seg, self).__init__()

        self.cov1 = nn.Sequential(
            nn.Conv2d(input_nc, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2)
        )

        self.cov2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels, base_channels * 3, kernel_size=4, stride=2, padding=1, bias=False),  # 从2倍增加到3倍
            nn.BatchNorm2d(base_channels * 3),
            nn.LeakyReLU(0.)
        )

        self.cov3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 3, base_channels * 6, kernel_size=4, stride=2, padding=1, bias=False),  # 从4倍增加到6倍
            nn.BatchNorm2d(base_channels * 6),
            nn.LeakyReLU(0.2)
        )

        self.cen = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 6, base_channels * 10, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 10, base_channels * 6, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 6),
            nn.Dropout(0.5)
        )

        self.dov3 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 12, base_channels * 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 3),
            nn.Dropout(0.5)
        )

        self.dov2 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 6, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.Dropout(0.5)
        )

        self.dov1 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels * 3, output_nc, kernel_size=4, stride=2, padding=1, bias=False),  # 拼接后通道调整
            nn.Tanh()
        )

    def forward(self, x):
        cx1 = self.cov1(x)
        cx2 = self.cov2(cx1)
        cx3 = self.cov3(cx2)

        ce = self.cen(cx3)

        d3 = torch.cat([ce, cx3], 1)
        dx3 = self.dov3(d3)

        d2 = torch.cat([dx3, cx2], 1)
        dx2 = self.dov2(d2)

        d1 = torch.cat([dx2, cx1], 1)
        dx1 = self.dov1(d1)

        return dx1


class Generator_Ct_Feature(nn.Module):
    def __init__(self, t=2):
        super(Generator_Ct_Feature, self).__init__()
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = RRCNN_block(ch_in=3, ch_out=32, t=t)
        self.down2 = RRCNN_block(ch_in=32, ch_out=64, t=t)
        self.down3 = RRCNN_block(ch_in=64, ch_out=128, t=t)
        self.down4 = RRCNN_block(ch_in=128, ch_out=256, t=t)
        self.down5 = RRCNN_block(ch_in=256, ch_out=512, t=t)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.Maxpool1(x1))
        x3 = self.down3(self.Maxpool1(x2))
        x4 = self.down4(self.Maxpool1(x3))
        x5 = self.down5(self.Maxpool1(x4))
        return x5


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.generate_CommonAndDiff = Generator_com_diff()
        self.UnetGenerator_out = UnetGenerator(input_nc=6, output_nc=3)
        self.Seg_out = Seg(input_nc=3, output_nc=3)
        self.end = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        pet_commonFeature, guided_CT, merge = self.generate_CommonAndDiff(x)
        dx1 = self.UnetGenerator_out(merge)
        segment = self.Seg_out(x)
        output = self.end(torch.cat([dx1, segment], dim=1))
        return guided_CT, pet_commonFeature, output, segment


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2))
            # layers.append(nn.Dropout(0.5))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
            # nn.Sigmoid()
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)







