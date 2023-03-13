import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmyolo.registry import MODELS

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        norm_cfg = dict(type='BN', momentum=0.01, eps=1e-5, affine=True)
        self.conv = ConvModule(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size-1) // 2,
            norm_cfg=norm_cfg,
            act_cfg=None
        )
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale

@MODELS.register_module()
class TripletAttention(nn.Module):
    def __init__(self, in_channels, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()      # bs, c, h, w -> bs, h, c, w
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous() # bs, h, c, w -> bs, c, h, w
        
        x_perm2 = x.permute(0,3,2,1).contiguous()      # bs, c, h, w -> bs, w, h, c
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous() # bs, w, h, c -> bs, c, h, w
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    triplet = TripletAttention()
    output=triplet(input)
    print(output.shape)
    print(triplet)
    