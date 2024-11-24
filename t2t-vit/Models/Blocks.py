import torch
import torch.nn as nn
from .Transformer import Transformer,token_Transformer_fuse

class Depthwise_Seperable_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3) -> None:
        super(Depthwise_Seperable_Conv, self).__init__()

        self.depth = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            dilation=dilation,
            padding=dilation,
        )

        self.point = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            dilation=dilation,
        )

    def forward(self, x):
        return self.point(self.depth(x))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, juh = 3, pad = 1):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, juh, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, juh, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)



class MFFM(nn.Module):
    def __init__(self, inchannels=64):
        super(MFFM, self).__init__()
        self.d_conv1 = nn.Sequential(
            nn.Conv2d(inchannels * 3, inchannels, 3, 1, 1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True)
        )
        self.d_conv2 = nn.Sequential(
            nn.Conv2d(inchannels, inchannels, 3, 1, 1),
            nn.BatchNorm2d(inchannels),
            nn.ReLU(inplace=True)
        )
        self.Sp=SpatialAttention()
    def forward(self, x):
        x = self.Sp(x)*self.d_conv1(x)
        return self.d_conv2(x)

class TFM(nn.Module):
    def __init__(self,inchannel=256, outchannel=64):
        super(TFM, self).__init__()
        self.rgbmlp = nn.Sequential(
            nn.Linear(inchannel, outchannel),
            nn.GELU(),
            nn.Linear(outchannel, outchannel),
        )
        self.depthmlp = nn.Sequential(
            nn.Linear(inchannel, outchannel),
            nn.GELU(),
            nn.Linear(outchannel, outchannel),
        )
        self.transformer = Transformer(embed_dim=outchannel, depth=2, num_heads=4, mlp_ratio=3.)
        self.token_trans = token_Transformer_fuse(embed_dim=outchannel, depth=2, num_heads=4, mlp_ratio=3.)
    def forward(self, siameae_token):
        B, _, _ = siameae_token.shape
        rgb_token_16, depth_token_16 = siameae_token.chunk(2)
        rgb_token_16, depth_token_16 = self.transformer(self.rgbmlp(rgb_token_16), self.depthmlp(depth_token_16))
        token_16 = self.token_trans(rgb_token_16, depth_token_16)
        token_16 = token_16.transpose(1, 2).contiguous().reshape(B//2, 64, 14, 14)
        return token_16