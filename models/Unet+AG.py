import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Attention gate mechanism for selecting key features from encoder to decoder"""

    def __init__(self, g_channels, x_channels, int_channels):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, int_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, int_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: Decoder features; x: Corresponding encoder layer features
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # Encoder features after weighted selection


class ConvBlock(nn.Module):
    """Convolutional block in encoder: 3x3 convolution + batch normalization + ReLU"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    """Upsampling block in decoder: 1x1 convolution + upsampling"""

    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

    def forward(self, x):
        return self.up(x)


class Generator(nn.Module):
    """Generator based on improved UNet, with attention gates and skip connections"""

    def __init__(self, in_channels=3, out_channels=3, init_channels=64):
        super(Generator, self).__init__()
        # Encoder (contracting path)
        self.encoder1 = ConvBlock(in_channels, init_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = ConvBlock(init_channels, init_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = ConvBlock(init_channels * 2, init_channels * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = ConvBlock(init_channels * 4, init_channels * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck layer
        self.bottleneck = ConvBlock(init_channels * 8, init_channels * 16)

        # Decoder (expanding path) and attention gates
        self.upconv4 = UpConvBlock(init_channels * 16, init_channels * 8)
        self.att4 = AttentionGate(init_channels * 8, init_channels * 8, init_channels * 4)
        self.decoder4 = ConvBlock(init_channels * 16, init_channels * 8)

        self.upconv3 = UpConvBlock(init_channels * 8, init_channels * 4)
        self.att3 = AttentionGate(init_channels * 4, init_channels * 4, init_channels * 2)
        self.decoder3 = ConvBlock(init_channels * 8, init_channels * 4)

        self.upconv2 = UpConvBlock(init_channels * 4, init_channels * 2)
        self.att2 = AttentionGate(init_channels * 2, init_channels * 2, init_channels * 1)
        self.decoder2 = ConvBlock(init_channels * 4, init_channels * 2)

        self.upconv1 = UpConvBlock(init_channels * 2, init_channels)
        self.att1 = AttentionGate(init_channels, init_channels, init_channels // 2)
        self.decoder1 = ConvBlock(init_channels * 2, init_channels)

        # Output layer: Generate pseudo-CT
        self.out_conv = nn.Conv2d(init_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder feature extraction
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        # Bottleneck layer
        b = self.bottleneck(self.pool4(e4))

        # Decoder feature fusion (with attention mechanism)
        d4 = self.upconv4(b)
        a4 = self.att4(d4, e4)  # Attention-based encoder feature selection
        d4 = torch.cat([d4, a4], dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        a3 = self.att3(d3, e3)
        d3 = torch.cat([d3, a3], dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        a2 = self.att2(d2, e2)
        d2 = torch.cat([d2, a2], dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        a1 = self.att1(d1, e1)
        d1 = torch.cat([d1, a1], dim=1)
        d1 = self.decoder1(d1)

        # Generate pseudo-CT
        pseudo_ct = self.out_conv(d1)
        return pseudo_ct




# Verify network structure
if __name__ == "__main__":
    # Input: 256x256 NAC PET slice (3 channels)
    pet = torch.randn(1, 3, 256, 256)
    real_ct = torch.randn(1, 3, 256, 256)

    generator = Generator(in_channels=3, out_channels=3)


