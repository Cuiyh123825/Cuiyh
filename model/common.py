"""Shared building blocks: PatchMerging, conv blocks, and the final
patch-expand head. These are direct ports from KMUNet with minor
reorganization but no behavioral changes.
"""

import torch
import torch.nn as nn
from einops import rearrange

from .kan_layers import PatchmergingBlock


class PatchMerging2D(nn.Module):
    """(B, H, W, C) -> (B, H/2, W/2, 2C) via 2x2 concat + KAN reduction."""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = PatchmergingBlock(dim=4 * dim)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        shape_fix = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            shape_fix[0] = H // 2
            shape_fix[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        if shape_fix[0] > 0:
            x0 = x0[:, : shape_fix[0], : shape_fix[1], :]
            x1 = x1[:, : shape_fix[0], : shape_fix[1], :]
            x2 = x2[:, : shape_fix[0], : shape_fix[1], :]
            x3 = x3[:, : shape_fix[0], : shape_fix[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class Final_PatchExpand2D(nn.Module):
    """Upsample by `dim_scale` (default 4) used as the final stem-inverse."""

    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c",
            p1=self.dim_scale, p2=self.dim_scale, c=C // self.dim_scale,
        )
        x = self.norm(x)
        return x


class DownConvBlock(nn.Module):
    """Multiple Conv-LeakyReLU-Dropout blocks stacked in series."""

    def __init__(self, in_channels, out_channels, dropout_p, depths):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depths):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Dropout(dropout_p),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class CnnDownBlock(nn.Module):
    """ConvBlock + downsample. Returns (downsampled, pre-downsample-feature)."""

    def __init__(self, in_channels, out_channels, dropout_p, depths, downsample):
        super().__init__()
        self.downsample_mode = downsample
        self.DownConvBlock = DownConvBlock(in_channels, out_channels, dropout_p, depths)

        if downsample == "use_pooling":
            self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        elif downsample == "PatchMerging2D":
            self.down = PatchMerging2D(in_channels)
        else:
            raise ValueError(f"Unknown downsample mode {downsample}")

    def forward(self, x: torch.Tensor):
        x = self.DownConvBlock(x)
        x_before_downsample = x.clone()
        if self.downsample_mode == "use_pooling":
            x = self.down(x)
        else:
            x = self.down(x.permute(0, 2, 3, 1))
            x = x.permute(0, 3, 1, 2)
        return x, x_before_downsample
