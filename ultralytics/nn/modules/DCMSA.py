from typing import List, Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# 24.10.12 Multimodal Split Attention And Fusion YOLO

__all__ = ('MSFocus', 'MSConv', 'MSAAFC2f', 'MFSPPF', 'MFConcat', 'MSConcat', 'MSAAFC2f_head1', 'MSAAFC2f_head2',)  # , '', '', '', '', ''

from ultralytics.nn.modules import CBAM


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


class MSFocus2(nn.Module):  # 3/2xc1 2:1
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1=4, c2=64, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()  # c1=4 -> 3:1 -> Focus * 4 -> 12:4
        self.cout_x = c2 // 2
        self.cout_rgb = c2
        self.conv_rgb = nn.Conv2d(3, self.cout_rgb, k, 2, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.conv_focus_x = nn.Conv2d(4, self.cout_x, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(self.cout_x + self.cout_rgb)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x_rgb, x_x = x.split([3, 1], dim=1)
        x_rgb = self.conv_rgb(x_rgb)
        x_x = self.conv_focus_x(torch.cat((x_x[..., ::2, ::2], x_x[..., 1::2, ::2], x_x[..., ::2, 1::2], x_x[..., 1::2, 1::2]), 1))  # Focus
        return self.act(self.bn(torch.cat([x_rgb, x_x], dim=1)))


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
        return torch.cat([self.act(self.bn(self.conv_rgb(x_rgb))), self.act(self.bn2(self.conv_x(x_x)))], dim=1)


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


class DCCAtt(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1),
            nn.LayerNorm(channels),
            nn.ReLU(inplace=True)
        )
        # self.conv = nn.Conv2d(2, 1, 1)
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

    def __init__(self, c1, c2, n=1, shortcut=False, self_c=2, k=3, s=2, p=None, g=1, d=1, act=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
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
        satt, csatt = self.satt([x_rgb, pre_satt]) if self.cin_rgb > 64 else self.satt([x_x, pre_satt])  # 1 1 80 80  todo
        x_x = x_x + x_x * satt
        pre_catt = self.catt_fc(x[2]) if self.cin_rgb > 64 else None  # x[1].c=96 x[0].c=192
        catt, ccatt = self.catt([x_rgb, pre_catt])  # ccatt(connect_catt)
        x_rgb = x_rgb + x_rgb * catt * satt
        x = torch.cat([x_rgb, x_x], 1)

        y = list(self.cv_fusion1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))
        return [x_fusion, x_x, ccatt, csatt]


class DCMSA_Block_test(nn.Module):  # 去除深度信息
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, self_c=2, k=3, s=2, p=None, g=1, d=1, act=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
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
        satt, csatt = self.satt([torch.cat([x_rgb, x_x], 1), pre_satt]) if self.cin_rgb > 256 else self.satt([x_x, pre_satt])  # 1 1 80 80  todo
        x_x = x_x + x_x * satt
        pre_catt = self.catt_fc(x[2]) if self.cin_rgb > 64 else None  # x[1].c=96 x[0].c=192
        catt, ccatt = self.catt([x_rgb, pre_catt])  # ccatt(connect_catt)
        x_rgb = x_rgb + x_rgb * catt * satt
        x = torch.cat([x_rgb, x_x], 1)

        y = list(self.cv_fusion1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))
        return [x_fusion, x_x, ccatt, csatt]


class DCMSA_Block_test2(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, self_c=2, k=3, s=2, p=None, g=1, d=1, act=True, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
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
        # self.satt_fc = Conv(1, 1, 3, 2)
        # self.satt = CBAMSAtt(7)
        # self.catt_fc = nn.Sequential(
        #     nn.Linear(self.cin_rgb, self.cout_rgb),
        #     nn.LayerNorm(self.cout_rgb),
        #     nn.ReLU(inplace=True)
        # )
        # self.catt = CBAMCAtt(self.cout_rgb)
        self.cv_fusion1 = Conv(self.cout_rgb + self.cout_x, 2 * self.c, 1, 1)  # self.cin_rgb if useatt else
        self.cv_fusion2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m_fusion = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.satt = CBAMSAtt(7)
        self.catt = CBAMCAtt(self.cout_rgb)

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
        spatialatt = self.satt(x_x)
        x_x = x_x + x_x * spatialatt
        x_rgb = x_rgb + self.catt(x_rgb) * spatialatt
        x = torch.cat([x_rgb, x_x], 1)

        y = list(self.cv_fusion1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m_fusion)
        x_fusion = self.cv_fusion2(torch.cat(y, 1))
        return [x_fusion, x_x]  # , ccatt, csatt


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
