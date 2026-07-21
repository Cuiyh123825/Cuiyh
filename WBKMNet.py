"""WBKMNet — Wavelet-Boundary KAN-Mamba Network.

Integrates three orthogonal innovations on top of the KMUNet backbone:

1. ``WaveletKANPatchEmbed`` for (near-)lossless 4x downsampling with
   explicit low/high frequency band processing.
2. ``KGMSS2D`` selective scan with a learnable KAN gate that adaptively
   mixes the four scan directions per token.
3. ``KAN_SCA_Edge`` bridge that augments the original spatial-channel
   attention with a Sobel-KAN edge-gating pathway, paired with multi-
   scale auxiliary segmentation heads in the decoder for deep supervision.
"""

import torch
import torch.nn as nn

from .attention_bridge import KAN_SCA_Edge
from .decoder import WBKMNetDecoder
from .encoder import WBKMNetEncoder


class WBKMNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 2,
        depths=(1, 2, 6, 2),
        patch_size: int = 4,
        hidden_dims=(64, 128, 256, 512),
        bridge: bool = True,
        deep_supervision: bool = True,
        upsample: str = "InverseWaveletExpand",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.bridge = bridge
        self.deep_supervision = deep_supervision

        if self.bridge:
            self.KAN_SCA_Edge = KAN_SCA_Edge(list(hidden_dims), split_att="KAN")

        self.encoder = WBKMNetEncoder(
            in_channels=input_channels,
            hidden_dims=hidden_dims,
            depths=depths,
            patch_size=patch_size,
            norm_layer=nn.LayerNorm,
            drop_rate=(0.1, 0.2, 0.3, 0.4, 0.5),
            Downsample="PatchMerging2D",
        )
        self.decoder = WBKMNetDecoder(
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            depths=depths,
            patch_size=patch_size,
            upsample=upsample,
            deep_supervision=deep_supervision,
        )

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        if self.bridge:
            features = self.KAN_SCA_Edge(features)
        return self.decoder(features)


if __name__ == "__main__":
    # Smoke test — verify shapes only. Full training requires a CUDA build
    # with mamba-ssm installed; CPU runs will error inside KGMSS2D.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WBKMNet().to(device)
    x = torch.randn(2, 3, 256, 256, device=device)

    model.train()
    out = model(x)
    if isinstance(out, tuple):
        main, aux = out
        print("train mode -> main:", tuple(main.shape),
              " | aux:", [tuple(a.shape) for a in aux])
        assert main.shape == (2, 2, 256, 256)
        for a in aux:
            assert a.shape == (2, 2, 256, 256)
    else:
        print("train mode (deep_sup off) -> main:", tuple(out.shape))

    model.eval()
    with torch.no_grad():
        y = model(x)
    print("eval mode  -> main:", tuple(y.shape))
    assert y.shape == (2, 2, 256, 256)
