"""Innovation 1 — Wavelet-KAN PatchEmbed.

Replaces KMUNet's interval-sampling ``KAN_PatchEmbed`` with a learnable
(near-)lossless Haar wavelet decomposition. The 4x downsample is realised
as two consecutive 2x2 Haar DWTs, producing 16 wavelet sub-bands per
input channel. The low-frequency (LL) band is encoded by a KAN tuned for
semantics; the three high-frequency bands (LH/HL/HH) are encoded by a
KAN tuned for boundary signals. The two encoded streams are concatenated
along the channel dimension.

A dual inverse-wavelet expand module mirrors the operation in the decoder.
"""

import math

import torch
import torch.nn as nn

from .kan_layers import KANLinear, _kan_init_weights


def _haar_filters(in_channels: int, dtype=torch.float32):
    """Return the four Haar 2x2 analysis filters as a (4*C, 1, 2, 2) tensor
    suitable for a depthwise conv2d with groups=in_channels.
    """
    a = 0.5
    ll = torch.tensor([[a, a], [a, a]], dtype=dtype)
    lh = torch.tensor([[a, a], [-a, -a]], dtype=dtype)
    hl = torch.tensor([[a, -a], [a, -a]], dtype=dtype)
    hh = torch.tensor([[a, -a], [-a, a]], dtype=dtype)
    kernel = torch.stack([ll, lh, hl, hh], dim=0)             # (4, 2, 2)
    kernel = kernel.unsqueeze(1)                              # (4, 1, 2, 2)
    kernel = kernel.repeat(in_channels, 1, 1, 1)              # (4C, 1, 2, 2)
    return kernel


class HaarDWT2d(nn.Module):
    """Depthwise 2x2 Haar DWT. Output is interleaved as
    (LL_0, LH_0, HL_0, HH_0, LL_1, LH_1, ...) along the channel axis,
    producing ``4 * C`` channels at half spatial resolution.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer("weight", _haar_filters(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W). Depthwise conv with stride 2 -> (B, 4C, H/2, W/2)
        return nn.functional.conv2d(
            x, self.weight, bias=None, stride=2, padding=0, groups=self.in_channels
        )


class HaarIDWT2d(nn.Module):
    """Inverse depthwise 2x2 Haar transform. Inverse of :class:`HaarDWT2d`."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        # Note: for orthonormal Haar the inverse uses the same kernel scaled by 4.
        # We keep the analysis kernel (sum = 1 for LL) and recover via conv_transpose.
        self.register_buffer("weight", _haar_filters(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4C, H, W) -> (B, C, 2H, 2W). The grouped transpose works because
        # the four filters per channel are orthogonal, so summing the four upsampled
        # bands reconstructs the original signal up to a constant factor.
        return nn.functional.conv_transpose2d(
            x, self.weight, bias=None, stride=2, padding=0, groups=self.in_channels
        )


class _KANChannelMixer(nn.Module):
    """Apply a KANLinear over the channel dim of a (B, C, H, W) tensor."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.kan = KANLinear(in_channels, out_channels)
        self.apply(_kan_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # (B, C, H, W) -> (B*H*W, C) -> KAN -> (B, H, W, C_out) -> (B, C_out, H, W)
        flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        flat = self.kan(flat)
        return flat.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


class WaveletKANPatchEmbed(nn.Module):
    """4x downsampling stem with explicit frequency-band processing.

    Pipeline:
        (B, C_in, H, W)
        -> HaarDWT2d (2x)   -> (B,  4 C_in, H/2, W/2)
        -> HaarDWT2d (2x)   -> (B, 16 C_in, H/4, W/4)
        -> split LL vs HF   -> low (B, C_in, H/4, W/4), high (B, 15 C_in, H/4, W/4)
        -> KAN(low)         -> (B, low_out, H/4, W/4)
        -> KAN(high)        -> (B, high_out, H/4, W/4)
        -> cat + LayerNorm  -> (B, out_dim, H/4, W/4)
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 64,
                 low_ratio: float = 0.5, norm_layer=nn.LayerNorm):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim

        # The LL-of-LL band is at index 0 of the 16-band output (because the second
        # DWT applies the same band ordering to each of the 4 first-level bands).
        # Specifically: after two DWTs the 16 channels are grouped in blocks of 4
        # per first-level band, and within each block the order is LL, LH, HL, HH.
        # So absolute index 0 corresponds to LL(LL(x)). All other 15 channels carry
        # high-frequency information at some scale.

        self.dwt1 = HaarDWT2d(in_channels)            # -> 4 * C
        self.dwt2 = HaarDWT2d(in_channels * 4)        # -> 16 * C

        # Channel split: low band per input channel (C indices), the rest as high.
        # We pick index 0 of each 16-block as low-frequency.
        self._low_indices = [16 * c + 0 for c in range(in_channels)]
        all_indices = list(range(16 * in_channels))
        self._high_indices = [i for i in all_indices if i not in self._low_indices]

        low_out = max(int(round(out_dim * low_ratio)), 1)
        high_out = out_dim - low_out

        # Encode each branch
        self.low_kan = _KANChannelMixer(in_channels, low_out)
        self.high_kan = _KANChannelMixer(len(self._high_indices), high_out)

        self.norm = norm_layer(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, H, W)
        x = self.dwt1(x)
        x = self.dwt2(x)
        low = x[:, self._low_indices, :, :]
        high = x[:, self._high_indices, :, :]
        low = self.low_kan(low)
        high = self.high_kan(high)
        out = torch.cat([low, high], dim=1)
        # LayerNorm operates over the last dim, so permute to (B, H, W, C)
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        return out  # (B, H/4, W/4, out_dim) — matches downstream encoder layout


class InverseWaveletExpand(nn.Module):
    """2x upsampling using Haar IDWT with a KAN that maps C -> 4C/2 = 2C.

    Mirrors :class:`PatchExpand2D` in KMUNet:  (B, H, W, C) -> (B, 2H, 2W, C/2).
    """

    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        # Expand C -> 4C, then split into 4 bands of C channels each, then IDWT
        # reconstructs C channels at 2x resolution; finally we project to C/2 via
        # another KAN.
        self.expand_kan = _KANChannelMixer(dim, 4 * dim)
        self.idwt = HaarIDWT2d(dim)
        self.project_kan = _KANChannelMixer(dim, dim // 2)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C). Move to channel-first for conv ops.
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()              # (B, C, H, W)
        x = self.expand_kan(x)                              # (B, 4C, H, W)
        x = self.idwt(x)                                    # (B, C, 2H, 2W)
        x = self.project_kan(x)                             # (B, C/2, 2H, 2W)
        x = x.permute(0, 2, 3, 1).contiguous()              # (B, 2H, 2W, C/2)
        x = self.norm(x)
        return x


if __name__ == "__main__":
    # Quick sanity check
    embed = WaveletKANPatchEmbed(in_channels=3, out_dim=64)
    inp = torch.randn(2, 3, 256, 256)
    out = embed(inp)
    print("WaveletKANPatchEmbed:", inp.shape, "->", out.shape)
    expand = InverseWaveletExpand(dim=64)
    up = expand(out)
    print("InverseWaveletExpand:", out.shape, "->", up.shape)
