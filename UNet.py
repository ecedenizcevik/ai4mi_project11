#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Jose Dolz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from torch import Tensor

def random_weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


def conv_block(in_dim, out_dim, **kwconv):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim, **kwconv),
                             nn.BatchNorm2d(out_dim),
                             nn.PReLU())


def conv_block_asym(in_dim, out_dim, *, kernel_size: int):
        return nn.Sequential(nn.Conv2d(in_dim, out_dim,
                                       kernel_size=(kernel_size, 1),
                                       padding=(2, 0)),
                             nn.Conv2d(out_dim, out_dim,
                                       kernel_size=(1, kernel_size),
                                       padding=(0, 2)),
                             nn.BatchNorm2d(out_dim),
                             nn.PReLU())


class BottleNeck(nn.Module):
        def __init__(self, in_dim, out_dim, projectionFactor,
                     *, dropoutRate=0.01, dilation=1,
                     asym: bool = False, dilate_last: bool = False):
                super().__init__()
                self.in_dim = in_dim
                self.out_dim = out_dim
                mid_dim: int = in_dim // projectionFactor

                # Main branch

                # Secondary branch
                self.block0 = conv_block(in_dim, mid_dim, kernel_size=1)

                if not asym:
                        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=dilation, dilation=dilation)
                else:
                        self.block1 = conv_block_asym(mid_dim, mid_dim, kernel_size=5)

                self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

                self.do = nn.Dropout(p=dropoutRate)
                self.PReLU_out = nn.PReLU()

                if in_dim > out_dim:
                        self.conv_out = conv_block(in_dim, out_dim, kernel_size=1)
                elif dilate_last:
                        self.conv_out = conv_block(in_dim, out_dim, kernel_size=3, padding=1)
                else:
                        self.conv_out = nn.Identity()

        def forward(self, in_) -> Tensor:
                # Main branch
                # Secondary branch
                b0 = self.block0(in_)
                b1 = self.block1(b0)
                b2 = self.block2(b1)
                do = self.do(b2)

                output = self.PReLU_out(self.conv_out(in_) + do)

                return output

def he_weights_init(m):
        # Different initialization due to dealing with ReLU. He initialization is good for ReLU, something with vanishing gradients 
        # more stable learning.
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, a=0.25, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                        m.bias.data.fill_(0)
        elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)


def double_conv(in_dim, out_dim):
        # The level block of the paper: two conv_block, 3x3 and padded so that the spatial
        # size is preserved
        return nn.Sequential(conv_block(in_dim, out_dim, kernel_size=3, padding=1, bias=False),
                             conv_block(out_dim, out_dim, kernel_size=3, padding=1, bias=False))


class DownSampling(nn.Module):
        # BottleNeckDownSampling, keeping only the max pooling of its main branch and with its
        # projecting secondary branch replaced by the double convolution. The pooling indices
        # are not returned anymore, since the decoder up-convolves instead of unpooling.
        def __init__(self, in_dim, out_dim, *, dropoutRate: float = 0.0):
                super().__init__()

                # Main branch
                self.maxpool0 = nn.MaxPool2d(2)

                self.do = nn.Dropout(p=dropoutRate) if dropoutRate else nn.Identity()

                # Secondary branch
                self.block0 = double_conv(in_dim, out_dim)

        def forward(self, in_) -> Tensor:
                # Main branch
                maxpool_output = self.do(self.maxpool0(in_))

                # Secondary branch
                return self.block0(maxpool_output)


class UpSampling(nn.Module):
        # BottleNeckUpSampling with its MaxUnpool2d swapped for the paper's 2x2 up-convolution,
        # which halves the channels; the skip concatenation and the convolutions that follow
        # are the same idea as ENet's, minus the residual.
        def __init__(self, in_dim, out_dim):
                super().__init__()

                # Main branch
                self.unpool = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2)

                # Secondary branch: out_dim channels from the up-convolution, out_dim from the skip
                self.block0 = double_conv(out_dim * 2, out_dim)

        def forward(self, args) -> Tensor:
                # nn.Sequential cannot handle multiple parameters:
                in_, skip = args

                # Main branch
                up = self.unpool(in_)

                # An odd spatial size makes the up-convolution undershoot the skip by a pixel.
                # The paper crops the (larger) skip, we pad the (smaller) up-convolution, so
                # that the prediction keeps the resolution main.py expects.
                dh: int = skip.shape[2] - up.shape[2]
                dw: int = skip.shape[3] - up.shape[3]
                if dh or dw:
                        up = nn.functional.pad(up, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])

                # Secondary branch
                return self.block0(torch.cat((up, skip), dim=1))


class UNet(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, **kwargs):
                super().__init__()
                F: int = kwargs["factor"] if "factor" in kwargs else 2  # Channel growth per level
                K: int = kwargs["kernels"] if "kernels" in kwargs else 64  # n_kernels
                dropout: float = kwargs["dropout"] if "dropout" in kwargs else 0.5
                n_bottlenecks: int = kwargs["bottlenecks"] if "bottlenecks" in kwargs else 0

                # Initial convolution, input dim to K channels.
                self.conv0 = double_conv(in_dim, K)

                # Downsampling half, each level halves spatial size and doubles channels.
                self.down1 = DownSampling(K, K * F)
                self.down2 = DownSampling(K * F, K * F ** 2)
                # Dropout at end of the downsampling half as in the original paper.
                self.down3 = DownSampling(K * F ** 2, K * F ** 3, dropoutRate=dropout)
                self.down4 = DownSampling(K * F ** 3, K * F ** 4, dropoutRate=dropout)

                # Middle bottleneck
                self.middle = nn.Sequential(*[BottleNeck(K * F ** 4, K * F ** 4, F, dropoutRate=0.1)
                                              for _ in range(n_bottlenecks)])

                # Upsampling half
                self.up4 = UpSampling(K * F ** 4, K * F ** 3)
                self.up3 = UpSampling(K * F ** 3, K * F ** 2)
                self.up2 = UpSampling(K * F ** 2, K * F)
                self.up1 = UpSampling(K * F, K)

                # Final convolution: "a 1x1 convolution to map each 64-component feature vector
                # to the desired number of classes"
                self.final = nn.Conv2d(K, out_dim, kernel_size=1)

                print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) with {kwargs}")

        def forward(self, input):
                # Initial operations
                outputInitial = self.conv0(input)

                # Downsampling half, every level kept for its skip connection
                down1_out = self.down1(outputInitial)
                down2_out = self.down2(down1_out)
                down3_out = self.down3(down2_out)
                down4_out = self.down4(down3_out)

                # Middle operations (an empty Sequential, unless bottlenecks were asked for)
                middle_out = self.middle(down4_out)

                # Upsampling half
                up4_out = self.up4((middle_out, down3_out))
                up3_out = self.up3((up4_out, down2_out))
                up2_out = self.up2((up3_out, down1_out))
                up1_out = self.up1((up2_out, outputInitial))

                # Final convolution
                return self.final(up1_out)

        def init_weights(self, *args, **kwargs):
            self.apply(random_weights_init if kwargs.get("xavier") else he_weights_init)
