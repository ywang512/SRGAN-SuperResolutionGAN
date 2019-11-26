import math
import torch
from torch import nn


"""
TODO:
  - TEST: Do we need a tanh() at the end of Generator?
    - paper do not have tanh(), and rescale HR images to be [-1, 1]
    - some people use tanh(), and they rescale HR iamges to be [0, 1]
"""




class ResidualBlock(nn.Module):
    '''input.shape == output.shape'''
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        
        return x + residual


class UpsampleBLock(nn.Module):
    '''Upsample by a factor of up_scale'''
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    '''Generator MLP'''
    def __init__(self, scale_factor, num_residual_block=5):
        super(Generator, self).__init__()
        num_upsample_block = int(math.log(scale_factor, 2))
        
        ### block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        
        ### block 2: B residual blocks
        block2 = [ResidualBlock(64) for _ in range(num_residual_block)]
        self.block2 = nn.Sequential(*block2)
        
        ### block 3: followed by a residual sum
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        
        ### block 4: upsample block
        block4 = [UpsampleBLock(64, up_scale=2) for _ in range(num_upsample_block)]
        block4.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block4 = nn.Sequential(*block4)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block1 + block3)
        # not sure if we need a activation tanh() at the end
        # no activation in paper, but found in some implementation
        return (torch.tanh(block4) + 1) / 2


class Discriminator(nn.Module):
    '''Discriminator MLP'''
    def __init__(self, dense_choice=0):
        super(Discriminator, self).__init__()
        assert dense_choice in [0, 1, 2], "invalid input."
        
        ### incremental filters of VGG structure: [64, 128, 256, 512]
        net_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        ### dense choice
        if dense_choice == 0:
            # original paper version
            net_dense = nn.Sequential(
                nn.Flatten(), 
                nn.Linear(512*16*16, 1024), 
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 1)
            )
        elif dense_choice == 1:
            # add adaptive average pooled 
            net_dense = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),  # avg pool to N * 512 1 * 1
                nn.Conv2d(512, 1024, kernel_size=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(1024, 1, kernel_size=1)
            )
        elif dense_choice == 2:
            # only one fc layer
            net_dense = nn.Sequential(
                nn.Conv2d(512, 1, kernel_size=1, stride=1),
                nn.AvgPool2d(16)
            )
        net = nn.Sequential(*(list(net_conv)+list(net_dense)))
        self.net = net

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))