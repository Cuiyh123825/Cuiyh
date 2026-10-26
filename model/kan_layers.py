"""Kolmogorov-Arnold Network primitives.

Ported from the KMUNet reference implementation. The classes here are
purely architectural building blocks and contain no project-specific
logic, so they are unchanged in functionality but reorganized for
clarity.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class KANLinear(nn.Module):
    """Single Kolmogorov-Arnold linear layer with B-spline bases."""

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=(-1, 1),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (1.0 if self.enable_standalone_scale_spline else self.scale_spline)
                * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2 and x.size(1) == self.in_features
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output


def _kan_init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= max(m.groups, 1)
        m.weight.data.normal_(0, math.sqrt(2.0 / max(fan_out, 1)))
        if m.bias is not None:
            m.bias.data.zero_()


class _KANToken(nn.Module):
    """Wraps a KANLinear so it accepts (B, N, C) tokens."""

    def __init__(self, in_features, out_features, drop=0.0):
        super().__init__()
        self.fc = KANLinear(in_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(_kan_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        x = self.fc(x.reshape(B * N, C))
        return x.reshape(B, N, -1).contiguous()


class PatchbedLayer(_KANToken):
    """KAN that maps `in_features` channels -> 64 channels (used for stem)."""

    def __init__(self, in_features, drop=0.0):
        super().__init__(in_features=in_features, out_features=64, drop=drop)


class PatchmergingLayer(_KANToken):
    """KAN that halves the channel dim (used inside PatchMerging2D)."""

    def __init__(self, in_features, drop=0.0):
        super().__init__(in_features=in_features, out_features=in_features // 2, drop=drop)


class PatchexpandLayer(_KANToken):
    """KAN that doubles the channel dim (used inside PatchExpand2D)."""

    def __init__(self, in_features, out_features, drop=0.0):
        super().__init__(in_features=in_features, out_features=out_features, drop=drop)


class KANs(nn.Module):
    """Convenience wrapper: takes (B, H, W, C), applies PatchbedLayer."""

    def __init__(self, dim, drop=0.0):
        super().__init__()
        self.layer = PatchbedLayer(in_features=dim, drop=drop)
        self.apply(_kan_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.layer(x)
        return x.reshape(B, H, W, -1).contiguous()


class PatchmergingBlock(nn.Module):
    """(B, H, W, C) -> (B, H, W, C/2) via KAN."""

    def __init__(self, dim, drop=0.0):
        super().__init__()
        self.layer = PatchmergingLayer(in_features=dim, drop=drop)
        self.apply(_kan_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.layer(x)
        return x.reshape(B, H, W, -1).contiguous()


class PatchexpandBlock(nn.Module):
    """(B, H, W, C) -> (B, H, W, out_dim) via KAN."""

    def __init__(self, dim, out_dim, drop=0.0):
        super().__init__()
        self.out_dim = out_dim
        self.layer = PatchexpandLayer(in_features=dim, out_features=out_dim, drop=drop)
        self.apply(_kan_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        x = x.flatten(start_dim=1, end_dim=2)
        x = self.layer(x)
        return x.reshape(B, H, W, -1).contiguous()
