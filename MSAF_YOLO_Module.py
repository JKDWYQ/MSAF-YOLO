import torch
from torch import nn


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


class focus(nn.Module):  # 3/2xc1 2:1

    def __init__(self):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()  # c1=4 -> 3:1 -> Focus * 4 -> 12:4

    def forward(self, x_rgb):
        """Apply convolution, batch normalization and activation to input tensor."""
        return torch.cat((x_rgb[..., ::2, ::2], x_rgb[..., 1::2, ::2], x_rgb[..., ::2, 1::2], x_rgb[..., 1::2, 1::2]), 1)


class dconv(nn.Module):
    def __init__(self, c_0, c_1, k):
        super().__init__()
        self.dconv1_k = nn.Conv2d(c_0, c_1, kernel_size=(1, k), padding=(0, k // 2))
        self.bn = nn.BatchNorm2d(c_1)
        self.relu = nn.ReLU(inplace=True)
        self.dconvk_1 = nn.Conv2d(c_1, c_1, kernel_size=(k, 1), padding=(k // 2, 0))

    def forward(self, x):
        return self.dconvk_1(self.relu(self.bn(self.dconv1_k(x))))


class DCSAtt_RCA(nn.Module):  # 09107
    """Spatial-attention module."""

    def __init__(self, c=2, kernel_size=(3, 1, 5, 9)):  # 9 11
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.dconv0 = dconv(2, 1, kernel_size[0])
        self.dconv1 = dconv(1, 1, kernel_size[1])
        self.dconv2 = dconv(1, 1, kernel_size[2])
        self.dconv3 = dconv(1, 1, kernel_size[3])
        self.cv2 = nn.Conv2d(2, 1, 3, 1, 1, bias=False)
        self.fusion = nn.Conv2d(3, 1, 1, 1, 0, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out = torch.cat([torch.mean(x[0], 1, keepdim=True), torch.max(x[0], 1, keepdim=True)[0]], 1)
        out = self.pool_h(out) + self.pool_w(out)
        out = self.dconv0(out)
        Satt = out
        if x[1] is not None:
            Satt = self.cv2(torch.cat([out, x[1]], 1))
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

    def forward(self, x: torch.Tensor):
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
