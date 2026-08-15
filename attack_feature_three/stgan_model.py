import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SPADE(nn.Module):
    """SPatially-Adaptive DEnormalization."""
    def __init__(self, num_features, label_dim):
        super(SPADE, self).__init__()
        self.norm = nn.BatchNorm2d(num_features, affine=False)
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_dim, nhidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.mlp_gamma = nn.Conv2d(nhidden, num_features, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, num_features, kernel_size=3, padding=1)

    def forward(self, x, label):
        normalized = self.norm(x)
        label = F.interpolate(label, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(label)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta


class SPADEResidualBlock(nn.Module):
    """Residual block with SPADE normalization."""
    def __init__(self, dim_in, dim_out, label_dim):
        super(SPADEResidualBlock, self).__init__()
        self.actv = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = SPADE(dim_in, label_dim)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = SPADE(dim_out, label_dim)
        if dim_in != dim_out:
            self.skip = nn.Conv2d(dim_in, dim_out, kernel_size=1, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x, label):
        h = self.norm1(x, label)
        h = self.actv(h)
        h = self.conv1(h)
        h = self.norm2(h, label)
        h = self.actv(h)
        h = self.conv2(h)
        return self.skip(x) + h


class STGANGenerator(nn.Module):
    """STGAN generator with SPADE normalization."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(STGANGenerator, self).__init__()
        self.c_dim = c_dim
        curr_dim = conv_dim

        # Encoder
        self.enc0_conv = nn.Conv2d(3, curr_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.enc0_norm = SPADE(curr_dim, c_dim)

        self.enc1_conv = nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc1_norm = SPADE(curr_dim * 2, c_dim)
        curr_dim = curr_dim * 2

        self.enc2_conv = nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc2_norm = SPADE(curr_dim * 2, c_dim)
        curr_dim = curr_dim * 2

        # Bottleneck
        self.bottleneck = nn.ModuleList()
        for i in range(repeat_num):
            self.bottleneck.append(SPADEResidualBlock(curr_dim, curr_dim, c_dim))

        # Decoder
        self.dec0_conv = nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec0_norm = SPADE(curr_dim // 2, c_dim)
        curr_dim = curr_dim // 2

        self.dec1_conv = nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False)
        self.dec1_norm = SPADE(curr_dim // 2, c_dim)
        curr_dim = curr_dim // 2

        self.dec2_conv = nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False)
        self.dec2_tanh = nn.Tanh()

    def forward(self, x, c):
        # Expand condition vector to spatial label map.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))

        feature_maps = []

        # Encoder
        h = self.enc0_conv(x)
        feature_maps.append(h)
        h = self.enc0_norm(h, c)
        feature_maps.append(h)
        h = F.relu(h, inplace=True)
        feature_maps.append(h)

        h = self.enc1_conv(h)
        feature_maps.append(h)
        h = self.enc1_norm(h, c)
        feature_maps.append(h)
        h = F.relu(h, inplace=True)
        feature_maps.append(h)

        h = self.enc2_conv(h)
        feature_maps.append(h)
        h = self.enc2_norm(h, c)
        feature_maps.append(h)
        h = F.relu(h, inplace=True)
        feature_maps.append(h)

        # Bottleneck
        for block in self.bottleneck:
            h = block(h, c)
            feature_maps.append(h)

        # Decoder
        h = self.dec0_conv(h)
        h = self.dec0_norm(h, c)
        h = F.relu(h, inplace=True)
        feature_maps.append(h)

        h = self.dec1_conv(h)
        h = self.dec1_norm(h, c)
        h = F.relu(h, inplace=True)
        feature_maps.append(h)

        h = self.dec2_conv(h)
        h = self.dec2_tanh(h)
        feature_maps.append(h)

        return h, feature_maps


class STGANDiscriminator(nn.Module):
    """PatchGAN discriminator for STGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(STGANDiscriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))
