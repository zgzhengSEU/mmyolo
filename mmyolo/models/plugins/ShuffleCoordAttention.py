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
                 groups=8):
        super(ShuffleCoordAttention, self).__init__()
        self.groups = groups
        mid_channels = in_channels // groups
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish(inplace=True)

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
    input=torch.randn(50,512,7,7)
    sca = ShuffleCoordAttention(in_channels=512)
    output=sca(input)
    print(output.shape)

    