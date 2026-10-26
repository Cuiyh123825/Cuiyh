"""WBKMNet encoder.

Mirrors :class:`KMUNet_Encoder` but swaps the patch-embed stem for
:class:`WaveletKANPatchEmbed`. The four downsample stages still use the
KMUNet ``CnnDownBlock`` + ``PatchMerging2D`` pattern.
"""

from typing import List

import torch
import torch.nn as nn

from .common import CnnDownBlock
from .wavelet_embed import WaveletKANPatchEmbed


class WBKMNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims=(64, 128, 256, 512),
        depths=(1, 2, 6, 2),
        patch_size: int = 4,
        drop_rate=(0.1, 0.2, 0.3, 0.4, 0.5),
        norm_layer=nn.LayerNorm,
        use_checkpoint: bool = False,
        Downsample: str = "PatchMerging2D",
    ):
        super().__init__()
        assert patch_size == 4, "WBKMNet currently fixes the stem stride at 4."
        self.use_checkpoint = use_checkpoint
        self.num_layers = len(depths)
        self.Downsample = Downsample

        # Innovation 1 — Wavelet-KAN PatchEmbed.
        # Stem output channels must match hidden_dims[0] so the first encoder
        # stage receives the expected width.
        self.wkan_patch_embed = WaveletKANPatchEmbed(
            in_channels=in_channels, out_dim=hidden_dims[0], norm_layer=norm_layer
        )

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(
                CnnDownBlock(
                    in_channels=hidden_dims[i_layer],
                    out_channels=hidden_dims[i_layer],
                    dropout_p=drop_rate[i_layer + 1],
                    depths=depths[i_layer],
                    downsample=Downsample if i_layer < self.num_layers else None,
                )
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        # Input is (B, C, H, W). Stem returns channel-last (B, H/4, W/4, dim).
        x = self.wkan_patch_embed(x)
        x = x.permute(0, 3, 1, 2)
        for layer in self.layers:
            x, x_feature = layer(x)
            outputs.append(x_feature)
        return outputs
