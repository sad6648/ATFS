import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    """GRU cell with convolutional operators. Core of the Selective Transfer Unit (STU).
    
    STGAN (Liu et al., CVPR 2019) uses STU to adaptively select and transform
    encoder features for transfer to the decoder, conditioned on the attribute
    difference vector.
    """
    def __init__(self, n_attrs, in_dim, out_dim, kernel_size=3):
        super(ConvGRUCell, self).__init__()
        self.n_attrs = n_attrs
        self.upsample = nn.ConvTranspose2d(
            in_dim * 2 + n_attrs, out_dim, 4, 2, 1, bias=False
        )
        self.reset_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1,
                      (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1,
                      (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Sigmoid()
        )
        self.hidden = nn.Sequential(
            nn.Conv2d(in_dim + out_dim, out_dim, kernel_size, 1,
                      (kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Tanh()
        )

    def forward(self, input, old_state, attr):
        n, _, h, w = old_state.size()
        attr = attr.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h, w))
        state_hat = self.upsample(torch.cat([old_state, attr], 1))
        r = self.reset_gate(torch.cat([input, state_hat], dim=1))
        z = self.update_gate(torch.cat([input, state_hat], dim=1))
        new_state = r * state_hat
        hidden_info = self.hidden(torch.cat([input, new_state], dim=1))
        output = (1 - z) * state_hat + z * hidden_info
        return output, new_state


class STGANGenerator(nn.Module):
    """STGAN Generator (Liu et al., CVPR 2019).
    
    Architecture: Encoder + Decoder with Selective Transfer Units (STU).
    The STU uses a ConvGRU to selectively transfer encoder features to the
    decoder based on the attribute difference vector (target - source).
    This enables precise attribute editing while preserving
    attribute-irrelevant details.
    
    Reference:
      Liu et al., "STGAN: A Unified Selective Transfer Network for Arbitrary
      Image Attribute Editing," CVPR 2019.
    """
    def __init__(self, conv_dim=64, c_dim=5, n_layers=5,
                 shortcut_layers=4, stu_kernel_size=3,
                 use_stu=True, one_more_conv=False):
        super(STGANGenerator, self).__init__()
        self.n_attrs = c_dim
        self.n_layers = n_layers
        self.shortcut_layers = min(shortcut_layers, n_layers - 1)
        self.use_stu = use_stu

        # Encoder: stride-2 conv layers with BN + LeakyReLU
        # Following Table A in the supplementary material
        self.encoder = nn.ModuleList()
        in_channels = 3
        for i in range(self.n_layers):
            out_ch = conv_dim * (2 ** i)
            self.encoder.append(nn.Sequential(
                nn.Conv2d(in_channels, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = out_ch

        # Selective Transfer Units (one per shortcut connection)
        if use_stu:
            self.stu = nn.ModuleList()
            for i in reversed(range(self.n_layers - 1 - self.shortcut_layers,
                                     self.n_layers - 1)):
                ch = conv_dim * (2 ** i)
                self.stu.append(ConvGRUCell(self.n_attrs, ch, ch, stu_kernel_size))

        # Decoder: stride-2 deconv layers with skip connections
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                # First decoder layer: concat encoder bottleneck + attr vector
                in_dim = conv_dim * (2 ** (self.n_layers - 1)) + self.n_attrs
                out_dim = conv_dim * (2 ** (self.n_layers - 1))
                self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True)
                ))
            elif i <= self.shortcut_layers:
                # Shortcut layers: concat decoder output + STU output
                in_dim = conv_dim * 3 * (2 ** (self.n_layers - 1 - i))
                out_dim = conv_dim * (2 ** (self.n_layers - 1 - i))
                self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True)
                ))
            elif i < self.n_layers - 1:
                in_dim = conv_dim * (2 ** (self.n_layers - i))
                out_dim = conv_dim * (2 ** (self.n_layers - 1 - i))
                self.decoder.append(nn.Sequential(
                    nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True)
                ))
            else:
                in_dim = conv_dim * 3 if self.shortcut_layers == self.n_layers - 1 else conv_dim * 2
                if one_more_conv:
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(in_dim, conv_dim // 4, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(conv_dim // 4),
                        nn.ReLU(inplace=True),
                        nn.ConvTranspose2d(conv_dim // 4, 3, 3, 1, 1, bias=False),
                        nn.Tanh()
                    ))
                else:
                    self.decoder.append(nn.Sequential(
                        nn.ConvTranspose2d(in_dim, 3, 4, 2, 1, bias=False),
                        nn.Tanh()
                    ))

    def forward(self, x, c):
        """Forward pass.
        
        Args:
            x: Input image [B, 3, H, W] in range [-1, 1].
            c: Attribute difference vector [B, c_dim] (target - source).
        Returns:
            output: Edited image [B, 3, H, W].
            feature_maps: List of intermediate feature maps for ATFS.
        """
        feature_maps = []

        # Encoder forward
        enc_features = []
        h = x
        for layer in self.encoder:
            h = layer(h)
            enc_features.append(h)
            feature_maps.append(h)

        # Bottleneck
        out = enc_features[-1]

        # First decoder layer: concat attribute vector with bottleneck
        n, _, h_b, w_b = out.size()
        attr = c.view((n, self.n_attrs, 1, 1)).expand((n, self.n_attrs, h_b, w_b))
        out = self.decoder[0](torch.cat([out, attr], dim=1))
        feature_maps.append(out)

        stu_state = enc_features[-1]

        # Shortcut layers with STU
        for i in range(1, self.shortcut_layers + 1):
            if self.use_stu:
                stu_out, stu_state = self.stu[i - 1](
                    enc_features[-(i + 1)], stu_state, c
                )
                feature_maps.append(stu_out)
                out = torch.cat([out, stu_out], dim=1)
                out = self.decoder[i](out)
                feature_maps.append(out)
            else:
                out = torch.cat([out, enc_features[-(i + 1)]], dim=1)
                out = self.decoder[i](out)
                feature_maps.append(out)

        # Non-shortcut decoder layers
        for i in range(self.shortcut_layers + 1, self.n_layers):
            out = self.decoder[i](out)
            feature_maps.append(out)

        return out, feature_maps


class STGANDiscriminator(nn.Module):
    """STGAN Discriminator with adversarial and attribute classification branches.
    
    Following Liu et al., CVPR 2019: D_adv distinguishes real vs. fake images,
    D_att predicts the attribute vector.
    """
    def __init__(self, image_size=128, attr_dim=10, conv_dim=64,
                 fc_dim=1024, n_layers=5):
        super(STGANDiscriminator, self).__init__()
        layers = []
        in_channels = 3
        for i in range(n_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, conv_dim * (2 ** i), 4, 2, 1),
                nn.InstanceNorm2d(conv_dim * (2 ** i),
                                  affine=True, track_running_stats=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))
            in_channels = conv_dim * (2 ** i)

        self.conv = nn.Sequential(*layers)
        feature_size = image_size // (2 ** n_layers)
        self.fc_adv = nn.Sequential(
            nn.Linear(conv_dim * (2 ** (n_layers - 1)) * (feature_size ** 2), fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, 1)
        )
        self.fc_att = nn.Sequential(
            nn.Linear(conv_dim * (2 ** (n_layers - 1)) * (feature_size ** 2), fc_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(fc_dim, attr_dim),
        )

    def forward(self, x):
        y = self.conv(x)
        y = y.view(y.size()[0], -1)
        logit_adv = self.fc_adv(y)
        logit_att = self.fc_att(y)
        return logit_adv, logit_att
