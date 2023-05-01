import torch
import torch.nn as nn
from mmyolo.registry import MODELS

@MODELS.register_module()
class ShuffleCoordAttention(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, 
                 in_channels, 
                 groups=8,
                 act_cfg='SiLU'):
        super(ShuffleCoordAttention, self).__init__()
        self.groups = groups
        mid_channels = in_channels // groups
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        if act_cfg == 'SiLU':
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = MODELS.build(act_cfg)
        self.conv_h = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)

        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape
        
        #group into subfeatures
        x = x.reshape(b * self.groups, -1, h, w) # bs * groups, c // groups, h, w
        
        identity = x
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2) # 1 x 7 -> 7 x 1
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        out = identity * a_w * a_h
        
        
        out = out.reshape(b, -1, h, w) # bs, c, h, w
        out = self.channel_shuffle(out, 2)
        
        return out


if __name__ == '__main__':
    input=torch.randn(1,512,20,20)
    groups = 32
    model = ShuffleCoordAttention(in_channels=512, groups=groups)
    output=model(input)
    # print(output.shape)
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    from fvcore.nn import flop_count_str
    flops = FlopCountAnalysis(model, input)
    print(f'input shape: {input.shape}')
    print(f'reduce: {groups}')
    print(flop_count_table(flops))
    # print(flop_count_str(flops))
    
    '''
    input shape: torch.Size([1, 512, 20, 20])
    reduce: 4
    | module          | #parameters or shape   | #flops   |
    |:----------------|:-----------------------|:---------|
    | model           | 49.792K                | 5.755M   |
    |  conv1          |  16.512K               |  2.621M  |
    |   conv1.weight  |   (128, 128, 1, 1)     |          |
    |   conv1.bias    |   (128,)               |          |
    |  bn1            |  0.256K                |  0.102M  |
    |   bn1.weight    |   (128,)               |          |
    |   bn1.bias      |   (128,)               |          |
    |  conv_h         |  16.512K               |  1.311M  |
    |   conv_h.weight |   (128, 128, 1, 1)     |          |
    |   conv_h.bias   |   (128,)               |          |
    |  conv_w         |  16.512K               |  1.311M  |
    |   conv_w.weight |   (128, 128, 1, 1)     |          |
    |   conv_w.bias   |   (128,)               |          |
    |  pool_h         |                        |  0.205M  |
    |  pool_w         |                        |  0.205M  |
    '''

    '''
    input shape: torch.Size([1, 512, 20, 20])
    reduce: 8
    | module          | #parameters or shape   | #flops   |
    |:----------------|:-----------------------|:---------|
    | model           | 12.608K                | 3.133M   |
    |  conv1          |  4.16K                 |  1.311M  |
    |   conv1.weight  |   (64, 64, 1, 1)       |          |
    |   conv1.bias    |   (64,)                |          |
    |  bn1            |  0.128K                |  0.102M  |
    |   bn1.weight    |   (64,)                |          |
    |   bn1.bias      |   (64,)                |          |
    |  conv_h         |  4.16K                 |  0.655M  |
    |   conv_h.weight |   (64, 64, 1, 1)       |          |
    |   conv_h.bias   |   (64,)                |          |
    |  conv_w         |  4.16K                 |  0.655M  |
    |   conv_w.weight |   (64, 64, 1, 1)       |          |
    |   conv_w.bias   |   (64,)                |          |
    |  pool_h         |                        |  0.205M  |
    |  pool_w         |                        |  0.205M  |
    '''

    '''
    input shape: torch.Size([1, 512, 20, 20])
    reduce: 16
    | module          | #parameters or shape   | #flops   |
    |:----------------|:-----------------------|:---------|
    | model           | 3.232K                 | 1.823M   |
    |  conv1          |  1.056K                |  0.655M  |
    |   conv1.weight  |   (32, 32, 1, 1)       |          |
    |   conv1.bias    |   (32,)                |          |
    |  bn1            |  64                    |  0.102M  |
    |   bn1.weight    |   (32,)                |          |
    |   bn1.bias      |   (32,)                |          |
    |  conv_h         |  1.056K                |  0.328M  |
    |   conv_h.weight |   (32, 32, 1, 1)       |          |
    |   conv_h.bias   |   (32,)                |          |
    |  conv_w         |  1.056K                |  0.328M  |
    |   conv_w.weight |   (32, 32, 1, 1)       |          |
    |   conv_w.bias   |   (32,)                |          |
    |  pool_h         |                        |  0.205M  |
    |  pool_w         |                        |  0.205M  |
    '''

    '''
    input shape: torch.Size([1, 512, 20, 20])
    reduce: 32
    | module          | #parameters or shape   | #flops   |
    |:----------------|:-----------------------|:---------|
    | model           | 0.848K                 | 1.167M   |
    |  conv1          |  0.272K                |  0.328M  |
    |   conv1.weight  |   (16, 16, 1, 1)       |          |
    |   conv1.bias    |   (16,)                |          |
    |  bn1            |  32                    |  0.102M  |
    |   bn1.weight    |   (16,)                |          |
    |   bn1.bias      |   (16,)                |          |
    |  conv_h         |  0.272K                |  0.164M  |
    |   conv_h.weight |   (16, 16, 1, 1)       |          |
    |   conv_h.bias   |   (16,)                |          |
    |  conv_w         |  0.272K                |  0.164M  |
    |   conv_w.weight |   (16, 16, 1, 1)       |          |
    |   conv_w.bias   |   (16,)                |          |
    |  pool_h         |                        |  0.205M  |
    |  pool_w         |                        |  0.205M  |
    '''