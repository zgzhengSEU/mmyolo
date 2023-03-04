import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmdet.utils import OptMultiConfig
from mmengine.model import BaseModule

from mmyolo.registry import MODELS


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

@MODELS.register_module()
class CoordAttention(nn.Module):
    def __init__(self, 
                 in_channels,  
                 reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        out_channels = in_channels
        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


# in_channels = torch.rand([2, 96, 56, 56])
# in_channels_dim, out_channels_dim = 96, 96
# reduction = 32

# coord_attention = CoordAttention(in_channels_dim, out_channels_dim, reduction=reduction)
# output = coord_attention(in_channels)
# print(output.shape)


# @MODELS.register_module()
# class CBAM(BaseModule):
#     """Convolutional Block Attention Module. arxiv link:
#     https://arxiv.org/abs/1807.06521v2.

#     Args:
#         in_channels (int): The input (and output) channels of the CBAM.
#         reduce_ratio (int): Squeeze ratio in ChannelAttention, the intermediate
#             channel will be ``int(channels/ratio)``. Defaults to 16.
#         kernel_size (int): The size of the convolution kernel in
#             SpatialAttention. Defaults to 7.
#         act_cfg (dict): Config dict for activation layer in ChannelAttention
#             Defaults to dict(type='ReLU').
#         init_cfg (dict or list[dict], optional): Initialization config dict.
#             Defaults to None.
#     """

#     def __init__(self,
#                  in_channels: int,
#                  reduce_ratio: int = 16,
#                  kernel_size: int = 7,
#                  act_cfg: dict = dict(type='ReLU'),
#                  init_cfg: OptMultiConfig = None):
#         super().__init__(init_cfg)
#         self.channel_attention = ChannelAttention(
#             channels=in_channels, reduce_ratio=reduce_ratio, act_cfg=act_cfg)

#         self.spatial_attention = SpatialAttention(kernel_size)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward function."""
#         out = self.channel_attention(x) * x
#         out = self.spatial_attention(out) * out
#         return out
