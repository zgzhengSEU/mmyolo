import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmdet.utils import OptMultiConfig
from mmengine.model import BaseModule

from mmyolo.registry import MODELS


@MODELS.register_module()
class CoordAttention(nn.Module):
    def __init__(self, 
                 in_channels,  
                 reduction=16):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        out_channels = in_channels
        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish(inplace=True)

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

if __name__ == '__main__':
    input=torch.randn(1,512,20,20)
    reduction = 32
    model = CoordAttention(in_channels=512, reduction=reduction)
    output=model(input)
    # print(output.shape)
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    from fvcore.nn import flop_count_str
    flops = FlopCountAnalysis(model, input)
    print(f'input shape: {input.shape}')
    print(f'reduce: {reduction}')
    print(flop_count_table(flops))
    # print(flop_count_str(flops))
    
    '''
    input shape: torch.Size([1, 512, 20, 20])
    reduce: 4
    | module          | #parameters or shape   | #flops   |
    |:----------------|:-----------------------|:---------|
    | model           | 0.198M                 | 5.678M   |
    |  conv1          |  65.664K               |  2.621M  |
    |   conv1.weight  |   (128, 512, 1, 1)     |          |
    |   conv1.bias    |   (128,)               |          |
    |  bn1            |  0.256K                |  25.6K   |
    |   bn1.weight    |   (128,)               |          |
    |   bn1.bias      |   (128,)               |          |
    |  conv_h         |  66.048K               |  1.311M  |
    |   conv_h.weight |   (512, 128, 1, 1)     |          |
    |   conv_h.bias   |   (512,)               |          |
    |  conv_w         |  66.048K               |  1.311M  |
    |   conv_w.weight |   (512, 128, 1, 1)     |          |
    |   conv_w.bias   |   (512,)               |          |
    |  pool_h         |                        |  0.205M  |
    |  pool_w         |                        |  0.205M  |
    '''
    
    '''
    input shape: torch.Size([1, 512, 20, 20])
    reduce: 8
    | module          | #parameters or shape   | #flops   |
    |:----------------|:-----------------------|:---------|
    | model           | 99.52K                 | 3.044M   |
    |  conv1          |  32.832K               |  1.311M  |
    |   conv1.weight  |   (64, 512, 1, 1)      |          |
    |   conv1.bias    |   (64,)                |          |
    |  bn1            |  0.128K                |  12.8K   |
    |   bn1.weight    |   (64,)                |          |
    |   bn1.bias      |   (64,)                |          |
    |  conv_h         |  33.28K                |  0.655M  |
    |   conv_h.weight |   (512, 64, 1, 1)      |          |
    |   conv_h.bias   |   (512,)               |          |
    |  conv_w         |  33.28K                |  0.655M  |
    |   conv_w.weight |   (512, 64, 1, 1)      |          |
    |   conv_w.bias   |   (512,)               |          |
    |  pool_h         |                        |  0.205M  |
    |  pool_w         |                        |  0.205M  |
    '''
    
    '''
    input shape: torch.Size([1, 512, 20, 20])
    reduce: 16
    | module          | #parameters or shape   | #flops   |
    |:----------------|:-----------------------|:---------|
    | model           | 50.272K                | 1.727M   |
    |  conv1          |  16.416K               |  0.655M  |
    |   conv1.weight  |   (32, 512, 1, 1)      |          |
    |   conv1.bias    |   (32,)                |          |
    |  bn1            |  64                    |  6.4K    |
    |   bn1.weight    |   (32,)                |          |
    |   bn1.bias      |   (32,)                |          |
    |  conv_h         |  16.896K               |  0.328M  |
    |   conv_h.weight |   (512, 32, 1, 1)      |          |
    |   conv_h.bias   |   (512,)               |          |
    |  conv_w         |  16.896K               |  0.328M  |
    |   conv_w.weight |   (512, 32, 1, 1)      |          |
    |   conv_w.bias   |   (512,)               |          |
    |  pool_h         |                        |  0.205M  |
    |  pool_w         |                        |  0.205M  |
    '''
    
    '''
    input shape: torch.Size([1, 512, 20, 20])
    reduce: 32
    | module          | #parameters or shape   | #flops   |
    |:----------------|:-----------------------|:---------|
    | model           | 25.648K                | 1.068M   |
    |  conv1          |  8.208K                |  0.328M  |
    |   conv1.weight  |   (16, 512, 1, 1)      |          |
    |   conv1.bias    |   (16,)                |          |
    |  bn1            |  32                    |  3.2K    |
    |   bn1.weight    |   (16,)                |          |
    |   bn1.bias      |   (16,)                |          |
    |  conv_h         |  8.704K                |  0.164M  |
    |   conv_h.weight |   (512, 16, 1, 1)      |          |
    |   conv_h.bias   |   (512,)               |          |
    |  conv_w         |  8.704K                |  0.164M  |
    |   conv_w.weight |   (512, 16, 1, 1)      |          |
    |   conv_w.bias   |   (512,)               |          |
    |  pool_h         |                        |  0.205M  |
    |  pool_w         |                        |  0.205M  |
    '''