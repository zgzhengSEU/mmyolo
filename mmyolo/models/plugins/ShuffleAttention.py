import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter

from mmyolo.registry import MODELS




@MODELS.register_module()
class ShuffleAttention(nn.Module):
    """Constructs a Channel Spatial Group module.

    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, 
                 in_channels, 
                 groups=8):
        super(ShuffleAttention, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, in_channels // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, in_channels // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, in_channels // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, in_channels // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(in_channels // (2 * groups), in_channels // (2 * groups))
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
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
        #channel_split
        x_0, x_1 = x.chunk(2, dim=1) # bs * groups, c // (2 * groups), h, w

        # channel attention
        x_channel = self.avg_pool(x_0) # bs * groups, c // (2 * groups), 1, 1
        x_channel = self.cweight * x_channel + self.cbias # bs * groups, c // (2 * groups), 1, 1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1) # bs * groups, c // (2 * groups), h, w
        x_spatial = self.sweight * x_spatial + self.sbias # bs * groups, c // (2 * groups), h, w
        x_spatial = x_1 * self.sigmoid(x_spatial) # bs * groups, c // (2 * groups), h, w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1) # bs * groups, c // groups, h, w
        out = out.reshape(b, -1, h, w) # bs, c, h, w

        out = self.channel_shuffle(out, 2)
        return out

if __name__ == '__main__':
    input=torch.randn(1,512,20,20)
    groups = 16
    model = ShuffleAttention(in_channels=512, groups=groups)
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
    