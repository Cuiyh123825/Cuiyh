"""WBKMNet decoder with multi-scale deep supervision (Innovation 3b).

The base structure mirrors KMUNet's decoder: three :class:`KGMVSSLayer_up`
stages followed by ``Final_PatchExpand2D + final_conv``. In addition,
each decoder stage emits a low-resolution prediction through a 1x1
convolution head; these auxiliary outputs are upsampled to the input
resolution and returned alongside the main prediction so a deep
supervision loss can be applied during training.

At inference time only the main prediction is returned, so the auxiliary
heads add zero cost outside of training.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import Final_PatchExpand2D
from .kgm_mamba import KGMVSSLayer_up


class _AuxHead(nn.Module):
    """1x1 conv + bilinear upsample to (H_target, W_target)."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        x = self.conv(x)
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


class WBKMNetDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dims=(64, 128, 256, 512),
        depths=(1, 2, 6, 2),
        patch_size: int = 4,
        upsample: str = "InverseWaveletExpand",
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.depths = depths
        self.deep_supervision = deep_supervision

        self.up1 = KGMVSSLayer_up(
            dim=hidden_dims[-1], depth=depths[-1], upsample=upsample
        )
        self.up2 = KGMVSSLayer_up(
            dim=hidden_dims[-2], depth=depths[-2], upsample=upsample
        )
        self.up3 = KGMVSSLayer_up(
            dim=hidden_dims[-3], depth=depths[-3], upsample=upsample
        )

        self.final_up = Final_PatchExpand2D(
            dim=hidden_dims[0], dim_scale=patch_size, norm_layer=nn.LayerNorm
        )
        self.final_conv = nn.Conv2d(hidden_dims[0] // patch_size, num_classes, 1)

        # Auxiliary heads for deep supervision (innovation 3b).
        # Each head consumes a decoder feature map at progressively higher
        # resolution: aux_head[0] sits after up1, aux_head[1] after up2,
        # aux_head[2] after up3 (just before the final stem-inverse expand).
        # Channel counts mirror the post-upsample widths:
        #   after up1: hidden_dims[-1] // 2 channels at the resolution of x2
        #   after up2: hidden_dims[-2] // 2 channels at the resolution of x1
        #   after up3: hidden_dims[-3] // 2 channels at the resolution of x0
        self.aux_heads = nn.ModuleList([
            _AuxHead(hidden_dims[-1] // 2, num_classes),
            _AuxHead(hidden_dims[-2] // 2, num_classes),
            _AuxHead(hidden_dims[-3] // 2, num_classes),
        ])

    def forward(self, features: List[torch.Tensor]):
        x0, x1, x2, x3 = features[0], features[1], features[2], features[3]

        d3 = self.up1(x3, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up3(d2, x0)

        x = d1.permute(0, 2, 3, 1)
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        main = self.final_conv(x)

        if not self.deep_supervision or not self.training:
            return main

        target_size = main.shape[-2:]
        aux_list = [
            self.aux_heads[0](d3, target_size),
            self.aux_heads[1](d2, target_size),
            self.aux_heads[2](d1, target_size),
        ]
        return main, aux_list
