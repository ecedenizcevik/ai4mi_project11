#!/usr/bin/env python3

# MIT License

# Copyright (c) 2026

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


class ViT(nn.Module):
    """Convolution-free ViT for segmentation (SETR-Naive style).

    The image is cut into non-overlapping patches, each patch becomes one token,
    a transformer encoder mixes the tokens globally, and a linear head predicts
    the p*p logits of each patch. No convolution anywhere: patch extraction is a
    reshape, the embedding is a matrix multiplication.
    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim

        self.patch_size: int = kwargs.get("patch_size", 16)
        self.dim: int = kwargs.get("dim", 128)
        self.depth: int = kwargs.get("depth", 4)
        self.img_size: int = kwargs.get("img_size", 256)
        mlp_ratio: int = kwargs.get("mlp_ratio", 4)
        dropout: float = kwargs.get("dropout", 0.1)

        # 32 channels per head is a common compromise: enough to be expressive,
        # small enough to keep several heads at low embedding dimensions.
        heads: int = kwargs.get("heads", max(1, self.dim // 32))
        assert self.dim % heads == 0, f"{self.dim=} not divisible by {heads=}"
        assert self.img_size % self.patch_size == 0

        p: int = self.patch_size
        self.grid: int = self.img_size // p
        n_tokens: int = self.grid ** 2

        self.patch_embed = nn.Linear(in_dim * p * p, self.dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, self.dim))

        layer = nn.TransformerEncoderLayer(d_model=self.dim,
                                           nhead=heads,
                                           dim_feedforward=self.dim * mlp_ratio,
                                           dropout=dropout,
                                           activation='gelu',
                                           batch_first=True,
                                           norm_first=True)  # pre-LN: stable without warmup
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.depth)
        self.norm = nn.LayerNorm(self.dim)
        self.head = nn.Linear(self.dim, out_dim * p * p)

        n_params: int = sum(q.numel() for q in self.parameters())
        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}) "
              f"with dim={self.dim}, depth={self.depth}, {heads=}, "
              f"patch={p}, {n_tokens=}, params={n_params}")

    def to_tokens(self, x: Tensor) -> Tensor:
        # [B, C, H, W] -> [B, N, C*p*p]
        B, C, H, W = x.shape
        p, g = self.patch_size, self.grid

        x = x.reshape(B, C, g, p, g, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # [B, g, g, C, p, p]
        return x.reshape(B, g * g, C * p * p)

    def to_image(self, x: Tensor) -> Tensor:
        # [B, N, K*p*p] -> [B, K, H, W]
        B, _, _ = x.shape
        p, g, K = self.patch_size, self.grid, self.out_dim

        x = x.reshape(B, g, g, K, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # [B, K, g, p, g, p]
        return x.reshape(B, K, g * p, g * p)

    def forward(self, input: Tensor) -> Tensor:
        assert input.shape[-2:] == (self.img_size, self.img_size), input.shape

        tokens = self.patch_embed(self.to_tokens(input))  # [B, N, D]
        tokens = tokens + self.pos_embed
        tokens = self.norm(self.encoder(tokens))

        return self.to_image(self.head(tokens))  # [B, K, H, W]

    def init_weights(self, *args, **kwargs):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)