#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026

"""Anisotropy-aware 3D U-Net.

Kernel and pooling sizes follow the voxel spacing rather than the array layout.
At the input spacing of (0.977, 0.977, 2.5) mm the axes differ by a factor 2.6,
so a symmetric 3x3x3 kernel would span 2.9 mm in-plane and 7.5 mm through-plane.
The first level therefore uses depth-1 kernels and pools in-plane only; after
that pooling the spacing is (1.95, 1.95, 2.5) mm, a ratio of 1.3, and isotropic
kernels become physically sensible. Rule taken from Isensee et al. (nnU-Net,
Nature Methods 2021).
"""

import torch
import torch.nn as nn
from torch import Tensor


def conv_block(in_dim: int, out_dim: int, *, depth_kernel: int) -> nn.Sequential:
    """Two convs, each followed by InstanceNorm and LeakyReLU.

    InstanceNorm rather than BatchNorm: patch-based 3D training runs at batch
    size 2, where per-batch statistics are too noisy to be useful.
    """
    k = (depth_kernel, 3, 3)
    p = (depth_kernel // 2, 1, 1)
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=k, padding=p, bias=False),
        nn.InstanceNorm3d(out_dim, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
        nn.Conv3d(out_dim, out_dim, kernel_size=k, padding=p, bias=False),
        nn.InstanceNorm3d(out_dim, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        K: int = kwargs.get("kernels", 16)          # base feature maps
        c1, c2, c3 = K, K * 2, K * 3

        # Level 1: depth-1 kernels, pooling in-plane only (spacing ratio 2.6)
        self.enc1 = conv_block(in_dim, c1, depth_kernel=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2))

        # Level 2 onward: near-isotropic, full 3D kernels and pooling
        self.enc2 = conv_block(c1, c2, depth_kernel=3)
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.bottleneck = conv_block(c2, c3, depth_kernel=3)

        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec2 = conv_block(c2 * 2, c2, depth_kernel=3)

        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = conv_block(c1 * 2, c1, depth_kernel=1)

        self.final = nn.Conv3d(c1, out_dim, kernel_size=1)

        n_params: int = sum(q.numel() for q in self.parameters())
        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) "
              f"with base={K}, params={n_params}")

    def forward(self, input: Tensor) -> Tensor:
        e1 = self.enc1(input)                        # [B, c1, D,   H,   W  ]
        e2 = self.enc2(self.pool1(e1))               # [B, c2, D,   H/2, W/2]
        b = self.bottleneck(self.pool2(e2))          # [B, c3, D/2, H/4, W/4]

        d2 = self.dec2(torch.cat((self.up2(b), e2), dim=1))
        d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))

        return self.final(d1)                        # [B, K, D, H, W]

    def init_weights(self, *args, **kwargs):
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm3d) and m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)