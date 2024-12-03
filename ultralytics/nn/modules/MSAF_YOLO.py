from typing import List, Any

import numpy as np
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from ultralytics.nn.modules import CBAM, RepConv
from ultralytics.nn.modules.block import C3k, C2f, Bottleneck_FADC

# 24.10.12 Multimodal Split Attention And Fusion YOLO

__all__ = ('MSFocus', 'MSConv', 'MSAAFC2f', 'MFSPPF', 'MFConcat', 'MSConcat', 'MSAAFC2f_head1', 'MSAAFC2f_head2',)  # , '', '', '', '', ''



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Bottleneck_att(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class MSFocus(nn.Module):  # 3/2xc1 2:1
    """
    split and Focus
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1=4, c2=64, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()  # c1=4 -> 3:1 -> Focus * 4 -> 12:4
        self.cout_x = c2 // 2
        self.cout_rgb = c2
        self.conv_focus_rgb = nn.Conv2d(12, self.cout_rgb, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_focus_d = nn.Conv2d(4, self.cout_x, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(self.cout_x + self.cout_rgb)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x_rgb, x_x = x.split([3, 1], dim=1)
        x_rgb = self.conv_focus_rgb(torch.cat((x_rgb[..., ::2, ::2], x_rgb[..., 1::2, ::2], x_rgb[..., ::2, 1::2], x_rgb[..., 1::2, 1::2]), 1))
        x_x = self.conv_focus_d(torch.cat((x_x[..., ::2, ::2], x_x[..., 1::2, ::2], x_x[..., ::2, 1::2], x_x[..., 1::2, 1::2]), 1))  # Focus
        return self.act(self.bn(torch.cat([x_rgb, x_x], dim=1)))


class MSConv2s(nn.Module):  # 3/2xc1 2:1
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1=4, c2=64, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()  # c1=4 -> 3:1 -> Focus * 4 -> 12:4
        self.cin_x = c1 if c1 > 4 else 1
        self.cin_rgb = c1 if c1 > 4 else 3
        self.cout_x = c2
        self.cout_rgb = c2

        self.conv_rgb = nn.Conv2d(self.cin_rgb, self.cout_rgb, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_x = nn.Conv2d(self.cin_x, self.cout_x, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(self.cout_rgb)
        self.bn2 = nn.BatchNorm2d(self.cout_x)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x_rgb, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)
        return [self.act(self.bn(self.conv_rgb(x_rgb))), self.act(self.bn2(self.conv_x(x_x)))]


class MFocus(nn.Module):  # 3/2xc1 2:1
    """
    split and Focus
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()  # c1=4 -> 3:1 -> Focus * 4 -> 12:4
        self.cout_x = c2 // 2
        self.cout_rgb = c2
        self.conv_rgbx = Conv(16, self.cout_rgb, k, s, p, g, act=act)
        self.conv_x = Conv(4, self.cout_x, k, s, p, g, act=act)

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x_rgb, x_x = x.split([3, 1], dim=1)
        x_rgbx = self.conv_rgbx(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        x_x = self.conv_x(torch.cat((x_x[..., ::2, ::2], x_x[..., 1::2, ::2], x_x[..., ::2, 1::2], x_x[..., 1::2, 1::2]), 1))  # Focus
        return torch.cat([x_rgbx, x_x], dim=1)


class MSConv(nn.Module):  # 3/2xc1 2:1
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.cin_x = c1 // 2 if c1 > 4 else 1
        self.cin_rgb = c1 if c1 > 4 else 3
        self.cout_x = c2 // 2
        self.cout_rgb = c2

        self.conv_rgb = nn.Conv2d(self.cin_rgb, self.cout_rgb, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_x = nn.Conv2d(self.cin_x, self.cout_x, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(self.cout_rgb + self.cout_x)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x_rgb, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)
        return self.act(self.bn(torch.cat([self.conv_rgb(x_rgb), self.conv_x(x_x)], dim=1)))


class MSConv2(nn.Module):  # 3/2xc1 2:1
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.cin_x = c1 // 2 if c1 > 4 else 1
        self.cin_rgb = c1 if c1 > 4 else 3
        self.cout_x = c2 // 2
        self.cout_rgb = c2

        self.conv_rgb = nn.Conv2d(self.cin_rgb, self.cout_rgb, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_x = nn.Conv2d(self.cin_x, self.cout_x, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(self.cout_rgb)
        self.bn2 = nn.BatchNorm2d(self.cout_x)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x_rgb, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)
        return [self.act(self.bn(self.conv_rgb(x_rgb))), self.act(self.bn2(self.conv_x(x_x)))]


class MSAAFC2f(nn.Module):  # 3/2xc1 2:1
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()  # self.cin_x = self.cin_rgb = c1
        self.cin_x = c1 // 2
        self.cin_rgb = c1
        self.c_rgb = int((self.cin_rgb + self.cin_x) * e)  # 改
        self.satt = CBAMSAtt(7)
        self.catt = CBAMCAtt(self.cin_rgb)
        self.cv_fusion1 = Conv(self.cin_rgb + self.cin_x, 2 * self.c_rgb, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c_rgb, self.cin_rgb, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(Bottleneck(self.c_rgb, self.c_rgb, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        x_rgb, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)

        spatialatt = self.satt(x_x)
        x_x = x_x + x_x * spatialatt
        x_rgb = x_rgb + self.catt(x_rgb) * spatialatt
        x = torch.cat([x_rgb, x_x], 1)

        y = list(self.cv_fusion1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))
        return torch.cat([x_fusion, x_x], dim=1)


class CBAMCAtt(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class CBAMSAtt(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))  # * x


class MFSPPF(nn.Module):  # 3/2xc1 2:1
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c1 = c1 // 2 * 3
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class MFConcat(nn.Module):  # 3/2xc1 2:1
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        # y0, y1, y2 = x[1].chunk(3, 1)
        return torch.cat([x[0], x[1]], self.d)


class MSConcat(nn.Module):  # 1xch 2:1
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        y0, y1, y2 = x[1].chunk(3, 1)
        return torch.cat([x[0], y0, y1], self.d)


class MSAAFC2f_head1(nn.Module):  # 不含深度信息
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()  # self.cin_x = self.cin_rgb = c1
        self.cin_x = 192
        self.cin_rgb = 960
        self.c_rgb = int(c2 * e)
        self.satt = CBAMSAtt(7)
        self.catt = CBAMCAtt(self.cin_rgb)
        self.cv_fusion1 = Conv(self.cin_rgb, 2 * self.c_rgb, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c_rgb, c2, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(Bottleneck(self.c_rgb, self.c_rgb, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):  # x0为上一层 x1为backbone
        """Forward pass through C2f layer."""
        x_fusion, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)

        spatialatt = self.satt(x_x)
        # x_x = x_x + x_x * spatialatt
        x_fusion = x_fusion + self.catt(x_fusion) * spatialatt
        # x = torch.cat([x_rgb, x_x], 1)

        y = list(self.cv_fusion1(x_fusion).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        return self.cv_fusion2(torch.cat(y, 1))


class MSAAFC2f_head2(nn.Module):  # 去除深度信息
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()  # self.cin_x = self.cin_rgb = c1
        self.cin_x = 288
        self.cin_rgb = 384
        self.c_rgb = int(c2 * e)
        self.satt = CBAMSAtt(7)
        self.catt = CBAMCAtt(self.cin_rgb)
        self.cv_fusion1 = Conv(self.cin_rgb, 2 * self.c_rgb, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c_rgb, c2, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(Bottleneck(self.c_rgb, self.c_rgb, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):  # x0为上一层 x1为backbone
        """Forward pass through C2f layer."""
        x_fusion, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)

        spatialatt = self.satt(x_x)
        # x_x = x_x + x_x * spatialatt
        x_fusion = x_fusion + self.catt(x_fusion) * spatialatt
        # x = torch.cat([x_rgb, x_x], 1)

        y = list(self.cv_fusion1(x_fusion).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        return self.cv_fusion2(torch.cat(y, 1))


class MSAAFC2f_CBAM(nn.Module):  # 去除深度信息
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()  # self.cin_x = self.cin_rgb = c1
        self.cin_x = c1 // 2
        self.cin_rgb = c1
        self.c_rgb = int((self.cin_rgb + self.cin_x) * e)  # 改
        self.satt = CBAMSAtt(7)
        self.catt = CBAMCAtt(self.cin_rgb + self.cin_x)
        self.cv_fusion1 = Conv(self.cin_rgb + self.cin_x, 2 * self.c_rgb, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c_rgb, c2, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(Bottleneck(self.c_rgb, self.c_rgb, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):  # x0为上一层 x1为backbone
        """Forward pass through C2f layer."""
        x_rgb, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)

        spatialatt = self.satt(x_x)
        x_x = x_x + x_x * spatialatt
        x = x + self.catt(x) * spatialatt

        y = list(self.cv_fusion1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))
        return torch.cat([x_fusion, x_x], dim=1)


class DCCAtt_block(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True)
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> List[Any]:
        b, c, _, _ = x[0].size()
        pre_Catt = x[1]
        gap = self.pool(x[0]).view(b, c)
        if pre_Catt is None:
            all_att = self.act(self.fc(gap.view(b, c, 1, 1))).view(b, c)
        else:
            all_att = self.act(self.fc(gap.view(b, c, 1, 1))).view(b, c)
            all_att = torch.cat([all_att.view(b, 1, 1, c), pre_Catt.view(b, 1, 1, c)], dim=1)
            all_att = self.conv(all_att).view(b, c)
            all_att = self.act(all_att)
        return [all_att.view(b, c, 1, 1), all_att.view(b, c)]


class DCSAtt_block(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.cv2 = nn.Conv2d(2, 1, 3, 1, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        pre_Satt = x[1]
        out = self.cv1(torch.cat([torch.mean(x[0], 1, keepdim=True), torch.max(x[0], 1, keepdim=True)[0]], 1))
        if pre_Satt is None:
            Satt = self.act(out)  # * x
        else:
            Satt = self.act(out)
            Satt = self.cv2(torch.cat([Satt, pre_Satt], 1))
            Satt = self.act(Satt)
        return [Satt, Satt]


class DCSAtt(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.cv2 = nn.Conv2d(2, 1, 3, 1, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        pre_Satt = x[1]
        out = self.cv1(torch.cat([torch.mean(x[0], 1, keepdim=True), torch.max(x[0], 1, keepdim=True)[0]], 1))
        if pre_Satt is None:
            Satt = self.act(out)  # * x
        else:
            Satt = self.cv2(torch.cat([out, pre_Satt], 1))
            Satt = self.act(Satt)
        return [Satt, out * Satt]


class CPCAttention(nn.Module):  # RepBlock

    def __init__(self, in_channels, out_channels,channelAttention_reduce=4):
        super().__init__()
        self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
        self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
        self.dconv1_11 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 11), padding=(0, 5), groups=in_channels)
        self.dconv11_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(11, 1), padding=(5, 0), groups=in_channels)
        self.dconv1_21 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 21), padding=(0, 10), groups=in_channels)
        self.dconv21_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(21, 1), padding=(10, 0), groups=in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):  # v1

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out


class dconv(nn.Module):
    def __init__(self, c_0, c_1, k):
        super().__init__()
        self.dconv1_k = nn.Conv2d(c_0, c_1, kernel_size=(1, k), padding=(0, k // 2))
        self.bn = nn.BatchNorm2d(c_1)
        self.relu = nn.ReLU(inplace=True)
        self.dconvk_1 = nn.Conv2d(c_1, c_1, kernel_size=(k, 1), padding=(k // 2, 0))

    def forward(self, x):
        return self.dconvk_1(self.relu(self.bn(self.dconv1_k(x))))


class DCSAtt_RCA2(nn.Module):
    """Spatial-attention module."""

    def __init__(self, c, band_kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        # self.dwconv_hw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.excite = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=1))
        # self.cv1 = nn.Conv2d(2, 1, 1, 1, 0, bias=False)
        self.cv2 = nn.Conv2d(2, 1, 3, 1, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        pre_Satt = x[1]
        out = torch.cat([torch.mean(x[0], 1, keepdim=True), torch.max(x[0], 1, keepdim=True)[0]], 1)
        # out = self.dwconv_hw(x[0])
        out = self.pool_h(out) + self.pool_w(out)  # .repeat(1,1,1,x_w.shape[-1])
        # out = self.pool_h(x[0]) + self.pool_w(x[0])  # .repeat(1,1,1,x_w.shape[-1])
        out = self.excite(out)  # .repeat(1,1,1,x_w.shape[-1])
        # out = self.cv1(torch.cat([torch.mean(out, 1, keepdim=True), torch.max(out, 1, keepdim=True)[0]], 1))
        if pre_Satt is None:
            Satt = self.act(out)  # [N, 1, C, 1]
        else:
            Satt = self.cv2(torch.cat([out, pre_Satt], 1))
            # Satt = self.cv2(out + pre_Satt)
            Satt = self.act(Satt)
        return [Satt, out * Satt]


class dconv(nn.Module):
    def __init__(self, c_0, c_1, k):
        super().__init__()
        self.dconv1_k = nn.Conv2d(c_0, c_1, kernel_size=(1, k), padding=(0, k // 2))
        self.bn = nn.BatchNorm2d(c_1)
        self.relu = nn.ReLU(inplace=True)
        self.dconvk_1 = nn.Conv2d(c_1, c_1, kernel_size=(k, 1), padding=(k // 2, 0))

    def forward(self, x):
        return self.dconvk_1(self.relu(self.bn(self.dconv1_k(x))))


class DCSAtt_RCA(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=(3, 1, 5, 9)):  # 9 11
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        # self.dwconv_hw = nn.Conv2d(c, c, 3, padding=1, groups=c)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.dconv0 = dconv(2, 1, kernel_size[0])
        self.dconv1 = dconv(1, 1, kernel_size[1])
        self.dconv2 = dconv(1, 1, kernel_size[2])
        self.dconv3 = dconv(1, 1, kernel_size[3])
        # self.cv1 = nn.Conv2d(2, 1, 3, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(2, 1, 3, 1, 1, bias=False)
        self.fusion = nn.Conv2d(3, 1, 1, 1, 0, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        pre_Satt = x[1]
        out = torch.cat([torch.mean(x[0], 1, keepdim=True), torch.max(x[0], 1, keepdim=True)[0]], 1)
        out = self.pool_h(out) + self.pool_w(out)
        out = self.dconv0(out)
        Satt = out
        if pre_Satt is not None:
            Satt = self.cv2(torch.cat([out, pre_Satt], 1))
        # Sobel = self.sobel_branch(Satt)
        dconv1 = self.dconv1(Satt)
        dconv2 = self.dconv2(Satt)
        dconv3 = self.dconv3(Satt)
        Satt = self.fusion(torch.cat([dconv1, dconv2, dconv3], 1))
        Satt = self.act(Satt)
        return [Satt, out * Satt]


class DCCAtt(nn.Module):
    def __init__(self, channels, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True)
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> List[Any]:
        b, c, _, _ = x[0].size()
        pre_Catt = x[1]
        gap = self.pool(x[0]).view(b, c)
        if pre_Catt is None:
            all_att = self.act(self.fc(gap.view(b, c, 1, 1))).view(b, c)
        else:
            all_att = torch.cat((gap.view(b, 1, 1, c), pre_Catt.view(b, 1, 1, c)), dim=1)
            all_att = self.conv(all_att).view(b, c)
            all_att = self.act(self.fc(all_att.view(b, c, 1, 1)))
        return [all_att.view(b, c, 1, 1), gap * all_att.view(b, c)]


class DCMSA(nn.Module):

    def __init__(self, c1, c2, DC=True, scale=2):
        super().__init__()
        self.PDCCA = nn.Sequential(nn.Linear(c1, c2), nn.LayerNorm(c2), nn.ReLU(inplace=True))
        self.ca = DCCAtt(c2)
        self.sa = DCSAtt_RCA(c2 // scale)
        self.PDCSA = Conv(1, 1, 3, 2)
        self.DC = DC
        self.rgb_to_x = nn.Conv2d(c2, c2 // scale, 1, 1, 0, bias=False)

    def forward(self, x):  # x[0]: x_rgb, x[1]: x_x, x[2]: pre Catt, x[3]: pre Satt
        PDCCA, PDCSA = [self.PDCCA(x[2]), self.PDCSA(x[3])] if self.DC else [None, None]

        RGBX_ca = x[0]  # x[0]+self.x_to_rgb(x[1])
        CA, DCCA = self.ca([RGBX_ca, PDCCA])
        RGB_CA = x[0] * CA
        RGBX_sa = torch.cat([RGB_CA, x[1]], 1) if self.DC else x[1]
        SA, DCSA = self.sa([RGBX_sa, PDCSA])

        x[0] = x[0] + RGB_CA * SA if self.DC else RGB_CA * SA
        x[1] = x[1] + x[1] * SA if self.DC else x[1] * SA

        return [x[0], x[1], DCCA, DCSA]


class PixelAttention_CGA(nn.Module):
    def __init__(self, dim):
        super(PixelAttention_CGA, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = rearrange(x2, 'b c t h w -> b (c t) h w')
        pattn2 = self.pa2(x2)
        return self.sigmoid(pattn2)


class DCMSA1(nn.Module):

    def __init__(self, c1, c2, DC=True):
        super().__init__()
        self.PDCCA = nn.Sequential(nn.Linear(c1, c2), nn.LayerNorm(c2), nn.ReLU(inplace=True))
        self.ca = DCCAtt(c2)
        self.sa = DCSAtt_RCA()
        self.PDCSA = Conv(1, 1, 3, 2)
        self.DC = DC

    def forward(self, x):  # x[0]: x_rgb, x[1]: x_x, x[2]: pre Catt, x[3]: pre Satt
        PDCCA, PDCSA = [self.PDCCA(x[2]), self.PDCSA(x[3])] if self.DC else [None, None]

        RGBX_ca = x[0]  # x[0]+self.x_to_rgb(x[1])
        CA, DCCA = self.ca([RGBX_ca, PDCCA])
        RGB_CA = x[0] * CA
        RGBX_sa = torch.cat([RGB_CA, x[1]], 1) if self.DC else x[1]
        SA, DCSA = self.sa([RGBX_sa, PDCSA])

        x[0] = x[0] + RGB_CA * SA if self.DC else RGB_CA * SA
        x[1] = x[1] + x[1] * SA if self.DC else x[1] * SA

        return [x[0], x[1], DCCA, DCSA]


class DCMSA2(nn.Module):

    def __init__(self, c1, c2, scale, DC=True):
        super().__init__()
        self.c_rgb = c2
        self.c_x = c2 // scale
        self.c_fusion = self.c_rgb + self.c_x
        self.PDCCA = nn.Sequential(nn.Linear(c1 + c1 // scale, self.c_fusion), nn.LayerNorm(self.c_fusion), nn.ReLU(inplace=True))
        self.ca = DCCAtt(self.c_fusion)
        self.PDCSA = Conv(1, 1, 3, 2)
        self.sa = DCSAtt()
        self.act = nn.Sigmoid()
        self.s = Parameter(torch.ones(1))
        self.rgb = Parameter(torch.ones(1))
        self.DC = DC
        self.rgb_to_x = nn.Conv2d(self.c_rgb, self.c_x, 1, 1, 0, bias=False)

    def forward(self, x):  # x[0]: x_rgb, x[1]: x_x, x[2]: pre Catt, x[3]: pre Satt
        PDCCA, PDCSA = [self.PDCCA(x[2]), self.PDCSA(x[3])] if self.DC else [None, None]

        RGBX_ca = torch.cat(x[:2], 1)  # x[0]+self.x_to_rgb(x[1])
        CA, DCCA = self.ca([RGBX_ca, PDCCA])
        RGB_CA, X_CA = (RGBX_ca * CA).split([self.c_rgb, self.c_x], 1)
        RGBX_sa = self.rgb * 0.5 * self.rgb_to_x(RGB_CA) + self.s * X_CA if self.DC else X_CA
        SA, DCSA = self.sa([RGBX_sa, PDCSA])

        x[0] = x[0] + RGB_CA * SA if self.DC else RGB_CA * SA
        x[1] = x[1] + X_CA * SA if self.DC else X_CA * SA

        return [x[0], x[1], DCCA, DCSA]


class MSAAFC2f_DCA(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()  # self.cin_x = self.cin_rgb = c1
        self.cin_x = c1 // 2
        self.cin_rgb = c1
        self.c_rgb = int((self.cin_rgb + self.cin_x) * e)  # 改
        self.satt = CBAMSAtt(7)
        self.catt = DCCAtt(self.cin_rgb)
        self.cv_fusion1 = Conv(self.cin_rgb + self.cin_x, 2 * self.c_rgb, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c_rgb, self.cin_rgb, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(Bottleneck(self.c_rgb, self.c_rgb, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        x_rgb, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)

        satt = self.satt(x_x)
        x_x = x_x + x_x * satt
        catt, ccatt = self.catt([x_rgb, None])  # ccatt(connect_catt)
        x_rgb = x_rgb + x_rgb * catt * satt
        # ccatt = ccatt * spatialatt  # todo
        x = torch.cat([x_rgb, x_x], 1)

        y = list(self.cv_fusion1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))
        return [torch.cat([x_fusion, x_x], dim=1), ccatt]


class C2f_2(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()  # self.cin_x = self.cin_rgb = c1
        self.cin_x = c1 // 2
        self.cin_rgb = c1
        # self.att = CBAM(self.cin_rgb + self.cin_x, 7)
        self.c_rgb = int((self.cin_rgb + self.cin_x) * e)  # 改
        self.cv_fusion1 = Conv(self.cin_rgb + self.cin_x, 2 * self.c_rgb, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c_rgb, self.cin_rgb, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(Bottleneck(self.c_rgb, self.c_rgb, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        x_rgb, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)
        y = list(self.cv_fusion1(x).chunk(2, 1))  # self.att(x)
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))
        return torch.cat([x_fusion, x_x], 1)


# MSAAFYOLO_DCA_backbone MSAAFC2f_DCA, MSAAFYOLO_DCA_Block
class MSAAFYOLO_DCA_Block(nn.Module):  # 去除深度信息
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, k=3, s=2, p=None, g=1, d=1, act=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()  # 0: c1=96 c2=192
        # MSConv
        self.cin_x = c1 // 2
        self.cin_rgb = c1
        self.cout_x = c2 // 2
        self.cout_rgb = c2
        self.conv_rgb = nn.Conv2d(self.cin_rgb, self.cout_rgb, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_x = nn.Conv2d(self.cin_x, self.cout_x, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # self.bn = nn.BatchNorm2d(self.cout_rgb + self.cout_x)
        self.bn_rgb = nn.BatchNorm2d(self.cout_rgb)
        self.bn_x = nn.BatchNorm2d(self.cout_x)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # MSAAFC2f_DCA
        self.c_rgb = int((self.cout_rgb) * e)  # todo 改 + self.cout_x
        self.satt_fc = Conv(1, 1, 3, 2)
        self.satt = DCSAtt(7)
        self.catt_fc = nn.Sequential(
            nn.Linear(self.cin_rgb, self.cout_rgb),
            nn.LayerNorm(self.cout_rgb),
            nn.ReLU(inplace=True)
        )
        self.catt = DCCAtt(self.cout_rgb)
        self.cv_fusion1 = Conv(self.cout_rgb + self.cout_x, 2 * self.c_rgb, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c_rgb, c2, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(Bottleneck(self.c_rgb, self.c_rgb, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):  # 0:1 144 160 160
        # Conv
        if self.cin_rgb > 64:
            x_rgb, x_x = x[0], x[1]
        else:
            x_rgb, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)  # 1 96 80 80, 1 48 80 80
        # x_rgb, x_x = self.act(self.bn(torch.cat([self.conv_rgb(x_rgb), self.conv_x(x_x)], dim=1))).split([self.cout_rgb, self.cout_x], dim=1)
        x_rgb = self.act(self.bn_rgb(self.conv_rgb(x_rgb)))
        x_x = self.act(self.bn_x(self.conv_x(x_x)))
        # C2f
        pre_satt = self.satt_fc(x[3]) if self.cin_rgb > 64 else None
        satt, csatt = self.satt([x_rgb, pre_satt]) if self.cin_rgb > 64 else self.satt([x_x, pre_satt])  # 1 1 80 80
        x_x = x_x + x_x * satt
        pre_catt = self.catt_fc(x[2]) if self.cin_rgb > 64 else None  # x[1].c=96 x[0].c=192
        catt, ccatt = self.catt([x_rgb, pre_catt])  # ccatt(connect_catt)
        x_rgb = x_rgb + x_rgb * catt * satt
        x = torch.cat([x_rgb, x_x], 1)

        y = list(self.cv_fusion1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))
        return [x_fusion, x_x, ccatt, csatt]


class DCMSA_Block(nn.Module):  # 去除深度信息
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, self_c=2, k=3, s=2, p=None, g=1, d=1, act=True,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()  # 0: c1=96 c2=192
        # MSConv
        self.cin_x = c1 // 2
        self.cin_rgb = c1
        self.cout_x = c2 // 2
        self.cout_rgb = c2
        self.conv_rgb = nn.Conv2d(self.cin_rgb, self.cout_rgb, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_x = nn.Conv2d(self.cin_x, self.cout_x, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # self.bn = nn.BatchNorm2d(self.cout_rgb + self.cout_x)
        self.bn_rgb = nn.BatchNorm2d(self.cout_rgb)
        self.bn_x = nn.BatchNorm2d(self.cout_x)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # MSAAFC2f_DCA
        self.c = int((self.cout_rgb + self.cout_x) * e) if self_c == 2 else int(self.cout_rgb * e)  # todo 改
        self.satt_fc = Conv(1, 1, 3, 2)
        self.satt = DCSAtt(7)
        self.catt_fc = nn.Sequential(
            nn.Linear(self.cin_rgb, self.cout_rgb),
            nn.LayerNorm(self.cout_rgb),
            nn.ReLU(inplace=True)
        )
        self.catt = DCCAtt(self.cout_rgb)
        self.cv_fusion1 = Conv(self.cout_rgb + self.cout_x, 2 * self.c, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):  # 0:1 144 160 160
        # Conv
        if self.cin_rgb > 64:
            x_rgb, x_x = x[0], x[1]
        else:
            x_rgb, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)  # 1 96 80 80, 1 48 80 80
        # x_rgb, x_x = self.act(self.bn(torch.cat([self.conv_rgb(x_rgb), self.conv_x(x_x)], dim=1))).split([self.cout_rgb, self.cout_x], dim=1)
        x_rgb = self.act(self.bn_rgb(self.conv_rgb(x_rgb)))
        x_x = self.act(self.bn_x(self.conv_x(x_x)))
        # C2f
        pre_satt = self.satt_fc(x[3]) if self.cin_rgb > 64 else None
        satt, csatt = self.satt([x_rgb, pre_satt]) if self.cin_rgb > 64 else self.satt([x_x, pre_satt])  # 1 1 80 80
        x_x = x_x + x_x * satt
        pre_catt = self.catt_fc(x[2]) if self.cin_rgb > 64 else None  # x[1].c=96 x[0].c=192
        catt, ccatt = self.catt([x_rgb, pre_catt])  # ccatt(connect_catt)
        x_rgb = x_rgb + x_rgb * catt * satt
        x = torch.cat([x_x, x_rgb], 1)

        y = list(self.cv_fusion1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))
        return [x_fusion, x_x, ccatt, csatt]


class DCMSA_Block_C3(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, self_c=2, k=3, s=2, p=None, d=1, act=True, g=1, e=0.5):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__()
        # MSConv
        self.cin_x = c1 // 2
        self.cin_rgb = c1
        self.cout_x = c2 // 2
        self.cout_rgb = c2
        self.conv_rgb = nn.Conv2d(self.cin_rgb, self.cout_rgb, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_x = nn.Conv2d(self.cin_x, self.cout_x, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # self.bn = nn.BatchNorm2d(self.cout_rgb + self.cout_x)
        self.bn_rgb = nn.BatchNorm2d(self.cout_rgb)
        self.bn_x = nn.BatchNorm2d(self.cout_x)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # MSAAFC2f_DCA
        self.c = int((self.cout_rgb + self.cout_x) * e) if self_c == 2 else int(self.cout_rgb * e)  # todo 改
        self.satt_fc = Conv(1, 1, 3, 2)
        self.satt = DCSAtt(7)
        self.catt_fc = nn.Sequential(
            nn.Linear(self.cin_rgb, self.cout_rgb),
            nn.LayerNorm(self.cout_rgb),
            nn.ReLU(inplace=True)
        )
        self.catt = DCCAtt(self.cout_rgb)
        self.cv1 = Conv((self.cout_rgb + self.cout_x), self.c, 1, 1)
        self.cv2 = Conv((self.cout_rgb + self.cout_x), self.c, 1, 1)
        self.cv3 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):  # 0:1 144 160 160
        # Conv
        if self.cin_rgb > 64:
            x_rgb, x_x = x[0], x[1]
        else:
            x_rgb, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)  # 1 96 80 80, 1 48 80 80
        # x_rgb, x_x = self.act(self.bn(torch.cat([self.conv_rgb(x_rgb), self.conv_x(x_x)], dim=1))).split([self.cout_rgb, self.cout_x], dim=1)
        x_rgb = self.act(self.bn_rgb(self.conv_rgb(x_rgb)))
        x_x = self.act(self.bn_x(self.conv_x(x_x)))
        # C2f
        pre_satt = self.satt_fc(x[3]) if self.cin_rgb > 64 else None
        satt, csatt = self.satt([x_rgb, pre_satt]) if self.cin_rgb > 64 else self.satt([x_x, pre_satt])  # 1 1 80 80  todo
        x_x = x_x + x_x * satt
        pre_catt = self.catt_fc(x[2]) if self.cin_rgb > 64 else None  # x[1].c=96 x[0].c=192
        catt, ccatt = self.catt([x_rgb, pre_catt])  # ccatt(connect_catt)
        x_rgb = x_rgb + x_rgb * catt * satt
        x = torch.cat([x_rgb, x_x], 1)

        x_fusion = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

        return [x_fusion, x_x, ccatt, csatt]


class SELayer(nn.Module):
    # SE module
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DCMSA_Block_test(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, self_c=2, k=3, s=2, p=None, g=1, d=1, act=True,
                 e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()  # 0: c1=96 c2=192
        # MSConv
        self.cin_x = c1 // 2
        self.cin_rgb = c1
        self.cout_x = c2 // 2
        self.cout_rgb = c2
        self.conv_rgb = nn.Conv2d(self.cin_rgb, self.cout_rgb, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_x = nn.Conv2d(self.cin_x, self.cout_x, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # self.bn = nn.BatchNorm2d(self.cout_rgb + self.cout_x)
        self.bn_rgb = nn.BatchNorm2d(self.cout_rgb)
        self.bn_x = nn.BatchNorm2d(self.cout_x)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        # MSAAFC2f_DCA
        # self.satt_fc = Conv(1, 1, 3, 2)
        # self.satt = CBAMSAtt(7)
        # self.catt_fc = nn.Sequential(
        #     nn.Linear(self.cin_rgb, self.cout_rgb),
        #     nn.LayerNorm(self.cout_rgb),
        #     nn.ReLU(inplace=True)
        # )
        # self.catt = CBAMCAtt(self.cout_rgb)
        # self.att = CBAM(self.cout_rgb + self.cout_x)
        # self.att = SELayer(self.cout_rgb + self.cout_x)
        # self.att = CPCAttention(self.cout_rgb + self.cout_x, self.cout_rgb + self.cout_x)
        self.x_to_rgb = Conv(self.cout_x, self.cout_rgb, 1, 1)  # todo 3, 1
        # self.fusion = iAFF(self.cout_rgb)  # self.cin_rgb if useatt else
        self.c = int((self.cout_rgb + self.cout_x) * e) if self_c == 2 else int(self.cout_rgb * e)  # todo 改
        self.cv_fusion1 = Conv(self.cout_rgb + self.cout_x if self_c == 2 else self.cout_rgb, 2 * self.c, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):  # 0:1 144 160 160
        # MSConv
        if self.cin_rgb > 64:
            x_rgb, x_x = x[0], x[1]
        else:
            x_rgb, x_x = x.split([self.cin_rgb, self.cin_x], dim=1)  # 1 96 80 80, 1 48 80 80
        x_rgb = self.act(self.bn_rgb(self.conv_rgb(x_rgb)))
        x_x = self.act(self.bn_x(self.conv_x(x_x)))
        # DCMSA
        # pre_satt = self.satt_fc(x[3]) if self.cin_rgb > 64 else None
        # satt, csatt = self.satt([x_rgb, pre_satt]) if self.cin_rgb > 64 else self.satt([x_x, pre_satt])  # 1 1 80 80
        # x_x = x_x + x_x * satt
        # pre_catt = self.catt_fc(x[2]) if self.cin_rgb > 64 else None  # x[1].c=96 x[0].c=192
        # catt, ccatt = self.catt([x_rgb, pre_catt])  # ccatt(connect_catt)
        # x_rgb = x_rgb + x_rgb * catt * satt
        # iAFF
        # residual = self.x_to_rgb(x_x)
        # x = self.fusion([x_rgb, residual])
        # C2f
        y = list(self.cv_fusion1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))
        return [x_fusion, x_x]  # , ccatt, csatt


class RGCSPELAN(nn.Module):
    def __init__(self, c1, c2, n=1, scale=0.5, e=0.5):
        super(RGCSPELAN, self).__init__()

        self.c = int(c1 * e)  # hidden channels
        self.mid = int(self.c * scale)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)

        self.cv3 = RepConv(self.c, self.mid, 3)
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))
        self.cv4 = Conv(self.mid, self.mid, 1)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))


class C2f_fusion(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, scale=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e * scale)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DCMSA_Block_test2(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, scale=1, k=3, s=2, p=None, g=1, d=1, e=0.5, act=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()  # 0: c1=96 c2=192
        # MSConv
        self.conv_rgb = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_x = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # DCMFA
        self.DC = True if c1 > 64 else False
        self.DCMSA = DCMSA(c1, c2, 4)
        # C2fusion
        self.rgb, self.x = Parameter(torch.ones(1)), Parameter(torch.ones(1))
        self.fusion = nn.Conv2d(c2, c2, 1, 1, autopad(1, p, d), groups=g, dilation=d, bias=False)
        self.c = int(c2 * e * scale)  # hidden channels
        self.cv1 = Conv(c2, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):  # 0:1 144 160 160
        # MSConv
        x_rgb, x_x = self.act(self.bn(self.conv_rgb(x[0]))), self.act(self.bn(self.conv_x(x[1])))
        # Deep Connection Attention
        x_rgb, x_x, DCCA, DCSA = self.DCMSA([x_rgb, x_x, x[2], x[3]]) if self.DC else self.DCMSA([x_rgb, x_x, None, None])
        # C2f
        y = list(self.cv1(self.fusion(self.rgb * x_rgb + self.x * x_x)).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        x_fusion = self.cv2(torch.cat(y, 1))
        return [x_fusion, x_x, DCCA, DCSA]  # [x_fusion, x_x]


class MFSPPF_DCA(nn.Module):  # 3/2xc1 2:1
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c1 = c1 // 2 * 3
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = torch.cat([x[0], x[1]], 1)
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class MSPPF_DCA(nn.Module):  # 3/2xc1 2:1
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c1 = c1
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x[0])
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class MSConcat_DCA(nn.Module):  # 3/2xc1 2:1
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        x_fusion = x[1][0]
        return torch.cat([x[0], x_fusion], self.d)


class DCMSA_C3k2(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, E=2, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__()
        self.cin_x = c1
        self.cin_rgb = c1
        self.cout_x = c2
        self.cout_rgb = c2

        self.satt_fc = Conv(1, 1, 3, 2)
        self.satt = DCSAtt(7)
        self.catt_fc = nn.Sequential(nn.Linear(self.cin_rgb // E, self.cin_rgb), nn.LayerNorm(self.cin_rgb), nn.ReLU(inplace=True)) if c1 >= 256 else None
        self.catt = DCCAtt(self.cin_rgb)

        self.c = int(self.cout_rgb * e)  # if self_c == 2 else int((self.cout_rgb + self.cout_x) * e) todo 改
        self.conv_x = Conv(self.cin_x, self.cout_x, 3, 1)  # self.cin_rgb if useatt else
        self.cv_fusion1 = Conv(self.cin_rgb + self.cin_x, 2 * self.c, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c, self.cout_rgb, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))
        # self.m_fusion2 = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):  # 0:1 144 160 160
        x_rgb, x_x = x[0], x[1]  # 1 96 80 80, 1 48 80 80
        pre_satt = None if self.cin_rgb <= 128 else self.satt_fc(x[2])
        pre_catt = None if self.cin_rgb <= 128 else self.catt_fc(x[3]) if self.catt_fc is not None else x[3]  # x[1].c=96 x[0].c=192

        satt, csatt = self.satt([x_rgb, pre_satt]) if self.cin_rgb > 128 else self.satt([x_x, pre_satt])  # 1 1 80 80  todo
        x_x = x_x + x_x * satt
        catt, ccatt = self.catt([x_rgb, pre_catt])  # ccatt(connect_catt)
        x_rgb = x_rgb + x_rgb * catt * satt

        x = torch.cat([x_rgb, x_x], 1)
        # x = x_rgb + x_x
        y = list(self.cv_fusion1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))
        x_x = self.conv_x(x_x)
        return [x_fusion, x_x, csatt, ccatt]


# two stream
class focus(nn.Module):  # 3/2xc1 2:1

    def __init__(self):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()  # c1=4 -> 3:1 -> Focus * 4 -> 12:4

    def forward(self, x_rgb):
        """Apply convolution, batch normalization and activation to input tensor."""
        return torch.cat((x_rgb[..., ::2, ::2], x_rgb[..., 1::2, ::2], x_rgb[..., ::2, 1::2], x_rgb[..., 1::2, 1::2]), 1)


class MSFocus2(nn.Module):  # 3/2xc1 2:1
    """
    split and Focus
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1=4, c2=64, k=1, s=1, scale=2, use=True, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()  # c1=4 -> 3:1 -> Focus * 4 -> 12:4
        self.focus = focus()
        self.x = c2 // scale
        self.rgb = c2
        self.cv1 = nn.Conv2d(c1, self.x, 3, 1, 1, groups=g, dilation=d, bias=False)
        self.cv2 = nn.Conv2d(c1 * 3, self.rgb, 3, 1, 1, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(self.rgb + self.x)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x_list = x.chunk(4, 1)
        x_rgb, x_x = [torch.cat(x_list[:3], 1), x_list[3]]  #
        x_rgb = self.cv2(self.focus(x_rgb))
        x_x = self.cv1(self.focus(x_x))
        return self.act(self.bn(torch.cat([x_rgb, x_x], dim=1))).split([self.rgb, self.x], 1)


class MSFocus1(nn.Module):  # 3/2xc1 2:1
    """
    split and Focus
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1=4, c2=64, k=1, s=1, use=True, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()  # c1=4 -> 3:1 -> Focus * 4 -> 12:4
        # self.cout_x = c2  # // 2
        # self.cout_rgb = c2
        self.focus = focus()
        self.use = use
        # if self.use:
        #     self.sobel_branch = SobelConv(24)
        #     self.pool_branch = nn.Sequential(
        #         nn.ZeroPad2d((0, 1, 0, 1)),
        #         nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)
        #     )
        #     self.branch_fusion = Conv(48, 24, 1, 1)
        #     self.fusion = nn.Conv2d(48, 24, 3, 1, 0, groups=g, dilation=d, bias=False)
        #
        self.cv2 = Conv(12, 48, 3, 1)
        self.conv_rgb = Conv(3, 48, 3, 2)
        self.fusion_rgb = nn.Conv2d(48, 48, 1, 1, 0, groups=g, dilation=d, bias=False)

        self.cv1 = Conv(4, 24, 3, 1)
        self.conv_x = Conv(1, 24, 3, 2)
        self.fusion_x = nn.Conv2d(24, 24, 1, 1, 0, groups=g, dilation=d, bias=False)

        self.bn = nn.BatchNorm2d(24 + 48)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x_list = x.chunk(4, 1)
        x_rgb, x_x = [torch.cat(x_list[:3], 1), x_list[3]]  #
        x_rgb = self.fusion_rgb(self.conv_rgb(x_rgb) + self.cv2(self.focus(x_rgb)))
        x_x = self.fusion_x(self.conv_x(x_x) + self.cv1(self.focus(x_x)))
        # if self.use:
        #     x_branch = self.branch_fusion(torch.cat([self.sobel_branch(x_x), self.pool_branch(x_x)], dim=1))
        #     x_x = self.fusion(torch.cat([x_x, x_branch], 1))
        # else:
        #     x_x = self.fusion(x_x)
        return self.act(self.bn(torch.cat([x_rgb, x_x], dim=1))).split([48, 24], 1)


class MSFocus_Module(nn.Module):  # 3/2xc1 2:1
    """
    split and Focus
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1=4, c2=64, k=1, s=1, scale=2, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()  # c1=4 -> 3:1 -> Focus * 4 -> 12:4
        self.cout_x = c2 // scale
        self.cout_rgb = c2
        self.focus = focus()
        self.conv_focus_x = nn.Conv2d(c1, self.cout_x, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_focus_rgb = nn.Conv2d(c1 * 3, self.cout_rgb, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(self.cout_x + self.cout_rgb)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x_list = x.chunk(4, 1)
        x_rgb, x_x = [torch.cat(x_list[:3], 1), x_list[3]]
        x_rgb = self.conv_focus_rgb(self.focus(x_rgb))
        x_x = self.conv_focus_x(self.focus(x_x))  # Focus
        return self.act(self.bn(torch.cat([x_rgb, x_x], dim=1))).split([self.cout_rgb, self.cout_x], 1)


class MSAF_Block1(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, scale=2, use_att=True, DC=True, k=3, s=2, p=None, g=1, d=1, e=0.5,
                 act=True):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()  # 0: c1=96 c2=192
        # MSConv
        self.conv_rgb = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_x = nn.Conv2d(c1 // scale, c2 // scale, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn_rgb = nn.BatchNorm2d(c2)
        self.bn_x = nn.BatchNorm2d(c2 // scale)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # DCMFA
        self.use_att = use_att
        self.DC = DC
        self.DCMSA = DCMSA1(c1, c2, self.DC)
        # C2fusion
        # self.rgb, self.x = Parameter(torch.ones(1)), Parameter(torch.ones(1))
        self.c = int((c2 + c2 // scale) * e)  # hidden channels
        self.cv1 = Conv((c2 + c2 // scale), 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):  # 0:1 144 160 160
        # MSConv
        x_rgb, x_x = self.act(self.bn_rgb(self.conv_rgb(x[0]))), self.act(self.bn_x(self.conv_x(x[1])))
        # Deep Connection Attention
        DCCA, DCSA = x[2:] if self.DC else [None, None]
        if self.use_att:
            x_rgb, x_x, DCCA, DCSA = self.DCMSA([x_rgb, x_x, DCCA, DCSA])
        # C2f
        y = list(self.cv1(torch.cat([x_x, x_rgb], dim=1)).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        x_fusion = self.cv2(torch.cat(y, 1))
        return [x_fusion, x_x, DCCA, DCSA]  # [x_fusion, x_x]


class MSAF_Block(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, scale=2, use_att=True, DC=True, e=0.5, k=3, s=2, p=None, g=1, d=1, act=True):
        super().__init__()
        # MSConv
        self.conv_rgb = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_x = nn.Conv2d(c1 // scale, c2 // scale, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn_rgb, self.bn_x = nn.BatchNorm2d(c2), nn.BatchNorm2d(c2 // scale)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # DCMSA
        self.use_att = use_att
        self.DC = DC
        if use_att:
            self.DCMSA = DCMSA(c1, c2, DC)
        # C2fusion
        self.c = int((c2 + c2 // scale) * e)  # hidden channels
        self.cv1 = Conv((c2 + c2 // scale), 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):  # 0:1 144 160 160
        # MSConv
        x_rgb, x_x = self.act(self.bn_rgb(self.conv_rgb(x[0]))), self.act(self.bn_x(self.conv_x(x[1])))
        # Deep Connection Attention
        DCCA, DCSA = x[2:] if self.DC else [None, None]
        if self.use_att:
            x_rgb, x_x, DCCA, DCSA = self.DCMSA([x_rgb, x_x, DCCA, DCSA])
        # C2f
        y = list(self.cv1(torch.cat([x_x, x_rgb], dim=1)).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        x_fusion = self.cv2(torch.cat(y, 1))
        return [x_fusion, x_x, DCCA, DCSA]


class MFSPPF_Module(nn.Module):  # 3/2xc1 2:1
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5, scale=3):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c1 = int((c2 + c2 // scale))
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = torch.cat([x[0], x[1]], 1)
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class DCMSConv(nn.Module):  # 3/2xc1 2:1
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x[0])))


class MSConv_x(nn.Module):  # 3/2xc1 2:1
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, index=0, p=None, g=1, d=1, act=True):
        super().__init__()
        self.i = index
        self.cin_x = c1
        self.cout_x = c2
        self.conv_x = nn.Conv2d(self.cin_x, self.cout_x, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(self.cout_x)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x_x = x[1] if self.i == 1 else x[0]  # x.split([self.cin_rgb, self.cin_x], dim=1)
        return self.act(self.bn(self.conv_x(x_x)))


class DCMSA_C3k2_rgbx(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, E=2, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__()
        self.cin_x = c1
        self.cin_rgb = c1
        self.cout_x = c2
        self.cout_rgb = c2

        self.satt_fc = Conv(1, 1, 3, 2)
        self.satt = DCSAtt(7)
        self.catt_fc = nn.Sequential(nn.Linear(self.cin_rgb // E, self.cin_rgb), nn.LayerNorm(self.cin_rgb), nn.ReLU(inplace=True)) if E == 2 else None
        self.catt = DCCAtt(self.cin_rgb)

        self.c = int(self.cout_rgb * e)  # if self_c == 2 else int((self.cout_rgb + self.cout_x) * e) todo 改
        self.cv_fusion1 = Conv(self.cin_rgb + self.cin_x, 2 * self.c, 1, 1)  # self.cin_rgb if useatt else
        self.cv_x1 = Conv(self.cin_x, 2 * self.c, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c, self.cout_rgb, 1)  # optional act=FReLU(c2)
        self.cv_x2 = Conv((2 + n) * self.c, self.cout_x, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))
        self.m_x = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):  # x = [x_x, [x_rgb, csatt, ccatt]]
        x_rgb, x_x = x[0], x[1]  # 1 96 80 80, 1 48 80 80
        pre_satt = None if self.cin_rgb <= 128 else self.satt_fc(x[2])
        pre_catt = None if self.cin_rgb <= 128 else self.catt_fc(x[3]) if self.catt_fc is not None else x[3]  # x[1].c=96 x[0].c=192

        satt, csatt = self.satt([x_rgb, pre_satt]) if self.cin_rgb > 128 else self.satt([x_x, pre_satt])  # 1 1 80 80  todo
        x_x = x_x + x_x * satt
        catt, ccatt = self.catt([x_rgb, pre_catt])  # ccatt(connect_catt)
        x_rgb = x_rgb + x_rgb * catt * satt

        x = torch.cat([x_x, x_rgb], 1)
        y = list(self.cv_fusion1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))

        y_x = list(self.cv_x1(x_x).chunk(2, 1))
        y_x.extend(m(y_x[-1]) for m in self.m_x)
        x_x = self.cv_x2(torch.cat(y_x, 1))

        return [x_fusion, x_x, csatt, ccatt]


class DCMSA_C3k2_test(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, E=2, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__()
        self.cin_x = c1
        self.cin_rgb = c1
        self.cout_x = c2
        self.cout_rgb = c2

        # self.satt_fc = Conv(1, 1, 3, 2)
        # self.satt = DCSAtt(7)
        # self.catt_fc = nn.Sequential(nn.Linear(self.cin_rgb // E, self.cin_rgb), nn.LayerNorm(self.cin_rgb), nn.ReLU(inplace=True)) if E == 2 else None
        # self.catt = DCCAtt(self.cin_rgb)

        self.c = int(self.cout_rgb * e)  # if self_c == 2 else int((self.cout_rgb + self.cout_x) * e) todo 改
        self.cv_fusion1 = Conv(self.cin_rgb + self.cin_x, 2 * self.c, 1, 1)  # self.cin_rgb if useatt else
        self.cv_x1 = Conv(self.cin_x, 2 * self.c, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c, self.cout_rgb, 1)  # optional act=FReLU(c2)
        self.cv_x2 = Conv((2 + n) * self.c, self.cout_x, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))
        self.m_x = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):  # x = [x_x, [x_rgb, csatt, ccatt]]
        x_rgb, x_x = x[0], x[1]  # 1 96 80 80, 1 48 80 80
        # pre_satt = None if self.cin_rgb <= 128 else self.satt_fc(x[2])
        # pre_catt = None  if self.cin_rgb <= 128 else self.catt_fc(x[3]) if self.catt_fc is not None else x[3]  # x[1].c=96 x[0].c=192

        # satt, csatt = self.satt([x_rgb, pre_satt]) if self.cin_rgb > 128 else self.satt([x_x, pre_satt])  # 1 1 80 80  todo
        # x_x = x_x + x_x * satt
        # catt, ccatt = self.catt([x_rgb, pre_catt])  # ccatt(connect_catt)
        # x_rgb = x_rgb + x_rgb * catt * satt

        y = list(self.cv_fusion1(torch.cat([x_rgb, x_x], 1)).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))

        y_x = list(self.cv_x1(x_x).chunk(2, 1))
        y_x.extend(m(y_x[-1]) for m in self.m_x)
        x_x = self.cv_x2(torch.cat(y_x, 1))
        return [x_fusion, x_x, None, None]


class MFSPPF2(nn.Module):  # 3/2xc1 2:1
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5, scale=3):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c1 = int((c2 + c2 // scale))
        c_ = c1 // 2
        # self.focus = focus()
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = torch.cat([x[0], x[1]], 1)
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class MSConcat2(nn.Module):

    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()
        assert fusion in ['concat', 'bifpn']
        self.fusion = fusion
        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4

    def forward(self, x):

        if self.fusion == 'concat':
            x_fusion = x[1][0]
            return torch.cat([x[0], x_fusion], self.d)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)


class ADD(nn.Module):  # 3/2xc1 2:1
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return x[0] + x[1]


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class CAA(nn.Module):
    def __init__(self, ch, h_kernel_size=11, v_kernel_size=11) -> None:
        super().__init__()

        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor * x


class Star_Block_CAA(Star_Block):
    def __init__(self, dim, mlp_ratio=3, drop_path=0):
        super().__init__(dim, mlp_ratio, drop_path)

        self.attention = CAA(mlp_ratio * dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(self.attention(x)))
        x = input + self.drop_path(x)
        return x


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape((B, -1, self.scale * H, self.scale * W))

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


class MSFocus3(nn.Module):  # 3/2xc1 2:1
    """
    split and Focus
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1=4, c2=64, k=1, s=1, scale=2, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.cout = c2 // scale
        self.focus = focus()
        self.d = nn.Conv2d(c1, self.cout, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.dgb = nn.Conv2d(3 * c1, self.cout, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.rdb = nn.Conv2d(3 * c1, self.cout, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.rgd = nn.Conv2d(3 * c1, self.cout, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(self.cout * 4)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        r, g, b, d = x.chunk(4, 1)
        dgb, rdb, rgd = [torch.cat([d, g, b], 1), torch.cat([r, d, b], 1), torch.cat([r, g, d], 1)]
        dgb, rdb, rgd, d = [self.focus(dgb), self.focus(rdb), self.focus(rgd), self.focus(d)]
        dgb, rdb, rgd, d = self.act(self.bn(torch.cat([self.dgb(dgb), self.rdb(rdb), self.rgd(rgd), self.d(d)], 1))).chunk(4, 1)
        return [dgb, rdb, rgd, d]


class DCMSA3(nn.Module):

    def __init__(self, c1, c2, DC=True):
        super().__init__()
        self.PDCCA0 = nn.Sequential(nn.Linear(c1, c2), nn.LayerNorm(c2), nn.ReLU(inplace=True))
        self.PDCCA1 = nn.Sequential(nn.Linear(c1, c2), nn.LayerNorm(c2), nn.ReLU(inplace=True))
        self.PDCCA2 = nn.Sequential(nn.Linear(c1, c2), nn.LayerNorm(c2), nn.ReLU(inplace=True))
        self.ca0 = DCCAtt(c2)
        self.ca1 = DCCAtt(c2)
        self.ca2 = DCCAtt(c2)
        self.PDCSA0 = Conv(1, 1, 3, 2)
        self.PDCSA1 = Conv(1, 1, 3, 2)
        self.PDCSA2 = Conv(1, 1, 3, 2)
        self.sa0 = DCSAtt()
        self.sa1 = DCSAtt()
        self.sa2 = DCSAtt()
        self.act = nn.Sigmoid()
        self.DC = DC

    def forward(self, x):  # [dgb, rdb, rgd, DCCA, DCSA]
        PDCCA0, PDCCA1, PDCCA2 = [self.PDCCA0(x[3][0]), self.PDCCA1(x[3][1]), self.PDCCA2(x[3][2])] if self.DC else [None, None, None]
        PDCSA0, PDCSA1, PDCSA2 = [self.PDCSA0(x[4][0]), self.PDCSA1(x[4][1]), self.PDCSA2(x[4][2])] if self.DC else [None, None, None]

        [CA0, DCCA0], [SA0, DCSA0] = self.ca0([x[0], PDCCA0]), self.sa0([x[0], PDCSA0])
        [CA1, DCCA1], [SA1, DCSA1] = self.ca1([x[1], PDCCA1]), self.sa1([x[1], PDCSA1])
        [CA2, DCCA2], [SA2, DCSA2] = self.ca2([x[2], PDCCA2]), self.sa2([x[2], PDCSA2])

        att0, att1, att2 = x[0] * CA0 * SA0, x[1] * CA1 * SA1, x[2] * CA2 * SA2
        dgb, rdb, rgd = [x[0] + att0, x[1] + att1, x[2] + att2] if self.DC else [att0, att1, att2]

        return [dgb, rdb, rgd, [DCCA0, DCCA1, DCCA2], [DCSA0, DCSA1, DCSA2]]


class MSAF_Block3(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, scale=2, use_att=True, k=3, s=2, p=None, g=1, d=1, e=0.5, act=True):
        super().__init__()  # 0: c1=96 c2=192
        # MSConv
        self.cin = c1 // scale
        self.cout = c2 // scale
        self.dgb = nn.Conv2d(self.cin, self.cout, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.rdb = nn.Conv2d(self.cin, self.cout, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.rgd = nn.Conv2d(self.cin, self.cout, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(self.cout)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        # attention module DCMFA
        self.use_att = use_att
        self.DC = True if c1 > 64 and use_att else False
        self.att = DCMSA3(self.cin, self.cout, self.DC)
        # C2fusion
        self.c = int(self.cout * e)  # hidden channels
        self.cv1_1 = Conv(self.cout * 2, 2 * self.c, 1, 1)
        self.cv2_1 = Conv(self.cout * 2, 2 * self.c, 1, 1)
        self.cv3_1 = Conv(self.cout * 2, 2 * self.c, 1, 1)
        self.m1 = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.m2 = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.m3 = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv1_2 = Conv((2 + n) * self.c, self.cout, 1)  # optional act=FReLU(c2)
        self.cv2_2 = Conv((2 + n) * self.c, self.cout, 1)  # optional act=FReLU(c2)
        self.cv3_2 = Conv((2 + n) * self.c, self.cout, 1)  # optional act=FReLU(c2)
        self.rgbd = nn.Conv2d(c1 if self.DC else self.cin, c1, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn_f = nn.BatchNorm2d(c1)
        self.cv1 = nn.Conv2d(c1 + self.cout * 3, self.cout * 2, 1, 1, 0, groups=g, dilation=d, bias=False)
        self.m = nn.ModuleList(Bottleneck(self.cout, self.cout, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cv2 = Conv((2 + n) * self.cout, c2, 1)  # optional act=FReLU(c2)

    def forward(self, x):  # 0:1 144 160 160
        # MSConv
        dgb, rdb, rgd = self.act(self.bn(self.dgb(x[0]))), self.act(self.bn(self.rdb(x[1]))), self.act(self.bn(self.rgd(x[2])))
        # Deep Connection Attention,
        if self.use_att:
            PDCCA, PDCSA = x[4:] if self.DC else [None, None]
            dgb, rdb, rgd, DCCA, DCSA = self.att([dgb, rdb, rgd, PDCCA, PDCSA])
        # C2f
        y1 = list(self.cv1_1(torch.cat([rdb, dgb], dim=1)).chunk(2, 1))  # , rgd
        y2 = list(self.cv2_1(torch.cat([rgd, rdb], dim=1)).chunk(2, 1))  # dgb,
        y3 = list(self.cv3_1(torch.cat([dgb, rgd], dim=1)).chunk(2, 1))  # rdb,
        y1.extend(m(y1[-1]) for m in self.m1)
        y2.extend(m(y2[-1]) for m in self.m2)
        y3.extend(m(y3[-1]) for m in self.m3)
        dgb, rdb, rgd = self.cv1_2(torch.cat(y1, 1)), self.cv2_2(torch.cat(y2, 1)), self.cv3_2(torch.cat(y3, 1))

        rgbd = self.act(self.bn_f(self.rgbd(x[3])))
        y = list(self.cv1(torch.cat([rgbd, dgb, rdb, rgd], 1)).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        rgbd = self.cv2(torch.cat(y, 1))

        return [dgb, rdb, rgd, rgbd] if not self.use_att else [dgb, rdb, rgd, rgbd, DCCA, DCSA]


class MFSPPF3(nn.Module):  # 3/2xc1 2:1
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5, scale=2):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        fusion = self.cv1(x[3])
        y1 = self.m(fusion)
        y2 = self.m(y1)
        return self.cv2(torch.cat((fusion, y1, y2, self.m(y2)), 1))


class MSConcat3(nn.Module):  # 3/2xc1 2:1
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        x_fusion = x[1][3]
        return torch.cat([x[0], x_fusion], self.d)


class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)

        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()

        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :, 0]


class EIEStem(nn.Module):
    def __init__(self, inc, hidc, ouc) -> None:
        super().__init__()

        self.conv1 = Conv(inc, hidc, 3, 2)
        self.sobel_branch = SobelConv(hidc)
        self.pool_branch = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)
        )
        self.conv2 = Conv(hidc * 2, hidc, 3, 2)
        self.conv3 = Conv(hidc, ouc, 1)

    def forward(self, x):
        input = x.clone()
        x = self.conv1(x)
        x = torch.cat([self.sobel_branch(x), self.pool_branch(x)], dim=1)
        x = self.conv2(x)
        x = self.conv3(x + input)
        return x


class EIEM(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()

        self.sobel_branch = SobelConv(inc)
        self.conv_branch = Conv(inc, inc, 3)
        self.conv1 = Conv(inc * 2, inc, 1)
        self.conv2 = Conv(inc, ouc, 1)

    def forward(self, x):
        x_sobel = self.sobel_branch(x)
        x_conv = self.conv_branch(x)
        x_concat = torch.cat([x_sobel, x_conv], dim=1)
        x_feature = self.conv1(x_concat)
        x = self.conv2(x_feature + x)
        return x


class CAFM(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(CAFM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=(1, 1, 1), bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim * 3, dim * 3, kernel_size=(3, 3, 3), stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=(1, 1, 1), bias=bias)
        self.fc = nn.Conv3d(3 * self.num_heads, 9, kernel_size=(1, 1, 1), bias=True)

        self.dep_conv = nn.Conv3d(9 * dim // self.num_heads, dim, kernel_size=(3, 3, 3), bias=True, groups=dim // self.num_heads, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.unsqueeze(2)
        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.squeeze(2)
        f_conv = qkv.permute(0, 2, 3, 1)
        f_all = qkv.reshape(f_conv.shape[0], h * w, 3 * self.num_heads, -1).permute(0, 2, 1, 3)
        f_all = self.fc(f_all.unsqueeze(2))
        f_all = f_all.squeeze(2)

        # local conv
        f_conv = f_all.permute(0, 3, 1, 2).reshape(x.shape[0], 9 * x.shape[1] // self.num_heads, h, w)
        f_conv = f_conv.unsqueeze(2)
        out_conv = self.dep_conv(f_conv)  # B, C, H, W
        out_conv = out_conv.squeeze(2)

        # global SA
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = out.unsqueeze(2)
        out = self.project_out(out)
        out = out.squeeze(2)
        output = out + out_conv

        return output


class CAFMFusion(nn.Module):
    def __init__(self, dim, heads):
        super(CAFMFusion, self).__init__()
        self.cfam = CAFM(dim, num_heads=heads)
        self.pa = PixelAttention_CGA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data[0][0], data[1]
        initial = x + y
        pattn1 = self.cfam(initial)
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result
