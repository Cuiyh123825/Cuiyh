"""Innovation 2 — KAN-Gated Multi-Path Selective Scan (KGM-SS2D).

KMUNet's :class:`SS2D` runs the 2D selective scan in four directions and
merges them by simple summation. We replace that fixed-weight merge with
a token-wise KAN gate: a tiny KANLinear computes a 4-dim softmax weight
per spatial token, letting the model adaptively bias the merge toward
the directions most aligned with the local lesion topology.

Everything else (per-direction projections, dt/A/B/C/D parameterisations,
the actual ``selective_scan_fn`` call) is preserved verbatim.
"""

import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import repeat
from timm.models.layers import DropPath

from .kan_layers import KANLinear, _kan_init_weights

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # type: ignore
except Exception:  # pragma: no cover
    selective_scan_fn = None  # type: ignore


class KGMSS2D(nn.Module):
    """KAN-gated 2D selective scan. Drop-in replacement for KMUNet's ``SS2D``."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = tuple(
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max,
                          dt_init_floor, **factory_kwargs)
            for _ in range(4)
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=4, merge=True)

        # ---- Innovation 2: KAN gate over the four scan directions ----
        # Operates per (B, H*W) token. We pool the post-conv feature down to a
        # compact summary (the channel mean) and let the KAN predict 4 logits.
        self.kan_gate = KANLinear(self.d_inner, 4)
        self._gate_init_done = False

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        self.apply(_kan_init_weights)

    # ---- helpers (verbatim from KMUNet) ----
    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
                 dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                   "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    # ---- core scan (per-direction outputs, gate fused in forward) ----
    def _scan_four(self, x: torch.Tensor):
        if selective_scan_fn is None:
            raise RuntimeError(
                "mamba_ssm.selective_scan_fn is not available. "
                "Install mamba-ssm to use KGMSS2D."
            )

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (B, K, D, L)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias, delta_softplus=True, return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )

        # All four are (B, d_inner, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = x_in.permute(0, 3, 1, 2).contiguous()
        x_in = self.act(self.conv2d(x_in))  # (B, d_inner, H, W)

        y1, y2, y3, y4 = self._scan_four(x_in)
        # y_i: (B, d_inner, L) where L = H*W
        L = H * W

        # ---- KAN gate (innovation 2) ----
        # Token-wise mean over channels -> (B, L). Replicate to (B, L, d_inner)?
        # We instead pool over channels to keep the gate cheap: (B, L, 1) is too
        # narrow for a KAN, so we use the full per-token channel vector (the
        # gating itself runs on the channel dim, not a pooled summary).
        x_token = x_in.flatten(2).transpose(1, 2).reshape(B * L, self.d_inner)
        g = self.kan_gate(x_token)              # (B*L, 4)
        g = F.softmax(g, dim=-1)
        g = g.view(B, L, 4).permute(0, 2, 1)    # (B, 4, L)

        stacked = torch.stack([y1, y2, y3, y4], dim=1)  # (B, 4, d_inner, L)
        y = (stacked * g.unsqueeze(2)).sum(dim=1)       # (B, d_inner, L)

        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class KGMVSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0.0,
        d_state: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = KGMSS2D(
            d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


class KGMVSSLayer(nn.Module):
    """Stack of KGMVSSBlocks; optional downsample at the end (channel-last layout)."""

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=16,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            KGMVSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)
        ])
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class KGMVSSLayer_up(nn.Module):
    """Decoder block: 1x1 conv to fuse skip + main, upsample, then KGMVSSBlocks."""

    def __init__(self, dim, depth, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
                 upsample="PatchExpand2D", use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.conv1x1 = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.blocks = nn.ModuleList([
            KGMVSSBlock(
                hidden_dim=dim // 2,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)
        ])
        # Late import to avoid circular dependency: WBKMNet imports from
        # wavelet_embed and kgm_mamba; wavelet_embed does not depend on us.
        from .wavelet_embed import InverseWaveletExpand
        if upsample == "InverseWaveletExpand":
            self.upsample = InverseWaveletExpand(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: (B, C, H, W) main stream (channel-first)
        # x2: (B, C', H', W') skip from encoder (channel-first)
        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.upsample(x1)
        x1 = x1.permute(0, 3, 1, 2)
        for blk in self.blocks:
            x_cat = torch.cat([x2, x1], dim=1)
            x_cat = self.conv1x1(x_cat)
            x1 = blk(x_cat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x1
