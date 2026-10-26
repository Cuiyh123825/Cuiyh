"""Innovation 3a — Edge-gated KAN_SCA bridge.

The base bridge (Channel_Att_Bridge + Spatial_Att_Bridge from KMUNet) is
preserved. On top we add a *Sobel-KAN edge-gating* pathway: a fixed
Sobel filter extracts a per-pixel edge response, a KAN learns how
strongly to amplify each skip feature based on the global average of
that response, and the resulting gate multiplies the skip feature in a
residual fashion. This biases the skip connection toward preserving
boundary-relevant detail for the decoder.
"""

import torch
import torch.nn as nn

from .kan_layers import KANLinear, _kan_init_weights


class Channel_Att_Bridge(nn.Module):
    """KAN-based channel attention spanning all four encoder scales."""

    def __init__(self, c_list, split_att="KAN"):
        super().__init__()
        c_list_sum = sum(c_list)
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        if split_att == "KAN":
            self.att1 = KANLinear(c_list_sum, c_list[0])
            self.att2 = KANLinear(c_list_sum, c_list[1])
            self.att3 = KANLinear(c_list_sum, c_list[2])
            self.att4 = KANLinear(c_list_sum, c_list[3])
        elif split_att == "fc":
            self.att1 = nn.Linear(c_list_sum, c_list[0])
            self.att2 = nn.Linear(c_list_sum, c_list[1])
            self.att3 = nn.Linear(c_list_sum, c_list[2])
            self.att4 = nn.Linear(c_list_sum, c_list[3])
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4):
        att = torch.cat(
            (self.avgpool(t1), self.avgpool(t2), self.avgpool(t3), self.avgpool(t4)), dim=1
        )
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att == "KAN":
            att = att.squeeze(1)
        att1 = self.sigmoid(self.att1(att))
        att2 = self.sigmoid(self.att2(att))
        att3 = self.sigmoid(self.att3(att))
        att4 = self.sigmoid(self.att4(att))
        if self.split_att == "KAN":
            att1 = att1.unsqueeze(1).transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(1).transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(1).transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(1).transpose(-1, -2).unsqueeze(-1).expand_as(t4)
        else:
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
        return att1, att2, att3, att4


class Spatial_Att_Bridge(nn.Module):
    """Avg+max pooling -> dilated 7x7 conv -> sigmoid."""

    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
            nn.Sigmoid(),
        )

    def forward(self, t1, t2, t3, t4):
        outs = []
        for t in (t1, t2, t3, t4):
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            outs.append(self.shared_conv2d(att))
        return outs[0], outs[1], outs[2], outs[3]


def _sobel_kernel(in_channels: int, dtype=torch.float32) -> torch.Tensor:
    """(2*C, 1, 3, 3) Sobel-x and Sobel-y kernels packed as a depthwise filter."""
    gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=dtype)
    gy = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=dtype)
    k = torch.stack([gx, gy], dim=0).unsqueeze(1)             # (2, 1, 3, 3)
    k = k.repeat(in_channels, 1, 1, 1)                        # (2*C, 1, 3, 3)
    return k


class SobelEdgeGate(nn.Module):
    """Edge-gating block applied per-scale.

    For an input ``x`` with ``C`` channels:
        1. Depthwise Sobel conv -> ``(B, 2C, H, W)`` edge response.
        2. Per-channel L2 -> ``(B, C, H, W)`` edge magnitude.
        3. Global average pool -> ``(B, C)`` summary.
        4. KAN -> sigmoid -> ``(B, C)`` per-channel gain.
        5. Residual modulation:  ``x' = x + g * x``.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.register_buffer("sobel", _sobel_kernel(channels))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.gate_kan = KANLinear(channels, channels)
        self.sigmoid = nn.Sigmoid()
        self.apply(_kan_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        e = nn.functional.conv2d(x, self.sobel, bias=None, stride=1, padding=1, groups=C)
        e = e.view(B, C, 2, H, W)
        edge_mag = torch.sqrt((e * e).sum(dim=2) + 1e-6)      # (B, C, H, W)
        summary = self.avgpool(edge_mag).view(B, C)           # (B, C)
        g = self.sigmoid(self.gate_kan(summary)).view(B, C, 1, 1)
        return x + g * x


class KAN_SCA_Edge(nn.Module):
    """KAN_SCA augmented with the SobelEdgeGate pathway at every scale."""

    def __init__(self, c_list, split_att="KAN"):
        super().__init__()
        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()
        self.edge_gates = nn.ModuleList([SobelEdgeGate(c) for c in c_list])

    def forward(self, t):
        r1, r2, r3, r4 = t[0], t[1], t[2], t[3]
        satt1, satt2, satt3, satt4 = self.satt(t[0], t[1], t[2], t[3])
        t1, t2, t3, t4 = satt1 * t[0], satt2 * t[1], satt3 * t[2], satt4 * t[3]

        r1_, r2_, r3_, r4_ = t1, t2, t3, t4
        t1, t2, t3, t4 = t1 + r1, t2 + r2, t3 + r3, t4 + r4

        catt1, catt2, catt3, catt4 = self.catt(t1, t2, t3, t4)
        t1, t2, t3, t4 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4

        # Innovation 3a — apply edge gating after the standard KAN_SCA output
        outs = []
        for i, (ti, ri_) in enumerate(zip((t1, t2, t3, t4), (r1_, r2_, r3_, r4_))):
            ti = ti + ri_
            ti = self.edge_gates[i](ti)
            outs.append(ti)
        return outs
