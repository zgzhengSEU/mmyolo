from mmyolo.registry import MODELS
import torch
import torch.nn.functional as F
import torch.nn as nn
from mmcv.ops.carafe import CARAFEPack
# from carafe import CARAFEPack
class SiLU(nn.Module):
    """export-friendly inplace version of nn.SiLU()"""

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    @staticmethod
    def forward(x):
        # clone is not supported with nni 2.6.1
        # result = x.clone()
        # torch.sigmoid_(x)
        return x * torch.sigmoid(x)


class HSiLU(nn.Module):
    """
        export-friendly inplace version of nn.SiLU()
        hardsigmoid is better than sigmoid when used for edge model
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    @staticmethod
    def forward(x):
        # clone is not supported with nni 2.6.1
        # result = x.clone()
        # torch.hardsigmoid(x)
        return x * torch.hardsigmoid(x)


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        # @ to do nn.SiLU 1.7.0
        # module = nn.SiLU(inplace=inplace)
        module = SiLU(inplace=inplace)
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'hsilu':
        module = HSiLU(inplace=inplace)
    elif name == 'identity':
        module = nn.Identity(inplace=inplace)
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act='silu'):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class shufflechannel_BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act='silu'):
        super().__init__()
        self.groups = groups
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.03, eps=0.001)
        self.act = get_activation(act, inplace=True)
        
    @staticmethod
    def shuffle_channels(x, groups):
        """shuffle channels of a 4-D Tensor"""
        batch_size, channels, height, width = x.size()
        assert channels % groups == 0
        channels_per_group = channels // groups
        # split into groups
        x = x.view(batch_size, groups, channels_per_group,
                height, width)
        # transpose 1, 2 axis
        x = x.transpose(1, 2).contiguous()
        # reshape into orignal
        x = x.view(batch_size, channels, height, width)
        return x
    
    def forward(self, x):
        return self.act(self.bn(self.conv(self.shuffle_channels(x, groups=self.groups))))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ASFF(nn.Module):

    def __init__(self,
                 level,
                 type='ASFF',
                 groups = 1,
                 asff_channel=2,
                 expand_kernel=3,
                 multiplier=1,
                 use_carafe=False,
                 use_softpool=False,
                 use_two_group_expand=False,
                 use_group_expand_nums=1,
                 act='silu'):
        """
        Args:
            level(int): the level of the input feature
            type(str): ASFF or ASFF_sim
            asff_channel(int): the hidden channel of the attention layer in ASFF
            expand_kernel(int): expand kernel size of the expand layer
            multiplier: should be the same as width in the backbone
        """
        super(ASFF, self).__init__()
        self.level = level
        self.type = type
        self.use_carafe = use_carafe
        self.use_softpool = use_softpool
        self.groups = groups
        self.use_two_group_expand = use_two_group_expand
        self.use_group_expand_nums = use_group_expand_nums
        if self.type == 'ASFF-X':
            self.dim = [
                int(1280 * multiplier),
                int(640 * multiplier),
                int(320 * multiplier),
                int(160 * multiplier)
            ]
        else:
            self.dim = [
                int(1024 * multiplier),
                int(512 * multiplier),
                int(256 * multiplier),
                int(128 * multiplier)
            ]
            
        Conv = BaseConv
        
        self.inter_dim = self.dim[self.level]

        if self.type == 'ASFF':
            if level == 0:
                self.stride_level_1 = Conv(int(512 * multiplier), self.inter_dim, 3, 2, groups=self.groups, act=act)
                self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2, groups=self.groups, act=act)
                self.stride_level_3 = Conv(int(128 * multiplier), self.inter_dim, 3, 2, groups=self.groups, act=act)
            elif level == 1:
                self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1, groups=self.groups, act=act)
                self.upsample_level_0 = CARAFEPack(channels=self.inter_dim, scale_factor=2) if self.use_carafe else nn.Upsample(scale_factor=2)
                self.stride_level_2 = Conv(int(256 * multiplier), self.inter_dim, 3, 2, groups=self.groups, act=act)
                self.stride_level_3 = Conv(int(128 * multiplier), self.inter_dim, 3, 2, groups=self.groups, act=act)
            elif level == 2:
                self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1, groups=self.groups, act=act)
                self.upsample_level_0 = CARAFEPack(channels=self.inter_dim, scale_factor=4) if self.use_carafe else nn.Upsample(scale_factor=4)
                self.compress_level_1 = Conv(int(512 * multiplier), self.inter_dim, 1, 1, groups=self.groups, act=act)
                self.upsample_level_1 = CARAFEPack(channels=self.inter_dim, scale_factor=2) if self.use_carafe else nn.Upsample(scale_factor=2)
                self.stride_level_3 = Conv(int(128 * multiplier), self.inter_dim, 3, 2, groups=self.groups, act=act)
            elif level == 3:
                self.compress_level_0 = Conv(int(1024 * multiplier), self.inter_dim, 1, 1, groups=self.groups, act=act)
                self.upsample_level_0 = CARAFEPack(channels=self.inter_dim, scale_factor=8) if self.use_carafe else nn.Upsample(scale_factor=8)
                self.compress_level_1 = Conv(int(512 * multiplier), self.inter_dim, 1, 1, groups=self.groups, act=act)
                self.upsample_level_1 = CARAFEPack(channels=self.inter_dim, scale_factor=4) if self.use_carafe else nn.Upsample(scale_factor=4)
                self.compress_level_2 = Conv(int(256 * multiplier), self.inter_dim, 1, 1, groups=self.groups, act=act)
                self.upsample_level_2 = CARAFEPack(channels=self.inter_dim, scale_factor=2) if self.use_carafe else nn.Upsample(scale_factor=2)
            else:
                raise ValueError('Invalid level {}'.format(level))
        else:
            if level == 0:
                pass
            elif level == 1:
                self.level_0_upsample = CARAFEPack(channels=self.dim[0], scale_factor=2) if self.use_carafe else nn.Upsample(scale_factor=2)
            elif level == 2:
                self.level_0_upsample = CARAFEPack(channels=self.dim[0], scale_factor=4) if self.use_carafe else nn.Upsample(scale_factor=4)
                self.level_1_upsample = CARAFEPack(channels=self.dim[1], scale_factor=2) if self.use_carafe else nn.Upsample(scale_factor=2)
            else:
                self.level_0_upsample = CARAFEPack(channels=self.dim[0], scale_factor=8) if self.use_carafe else nn.Upsample(scale_factor=8)
                self.level_1_upsample = CARAFEPack(channels=self.dim[1], scale_factor=4) if self.use_carafe else nn.Upsample(scale_factor=4)
                self.level_2_upsample = CARAFEPack(channels=self.dim[2], scale_factor=2) if self.use_carafe else nn.Upsample(scale_factor=2)
        
        if self.level in [0, 1]:
            if use_softpool:
                import softpool_cuda
                from SoftPool import SoftPool2d
                self.pool = SoftPool2d(kernel_size=2, stride=2)
            else:
                self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # add expand layer
        if self.use_group_expand_nums > 1:
            shuffleConv = []
            for i in range(self.use_group_expand_nums - 1):
                shuffleConv.append(shufflechannel_BaseConv(self.inter_dim, self.inter_dim, expand_kernel, 1, groups=self.groups, act=act))
            self.expand = nn.Sequential(
                Conv(self.inter_dim, self.inter_dim, expand_kernel, 1, groups=self.groups, act=act), 
                *shuffleConv)
        else:
            self.expand = Conv(self.inter_dim, self.inter_dim, expand_kernel, 1, groups=self.groups, act=act)
            
        self.weight_level_0 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)
        self.weight_level_1 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)
        self.weight_level_2 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)
        self.weight_level_3 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)

        self.weight_levels = Conv(asff_channel * 4, 4, 1, 1, act=act)
    
    def init_weights(self):
        """Initialize the weights of module."""
        super(ASFF, self).init_weights()
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()
                
    def focus_channels(self, x):
        # [b,c,h,w]->[b,c*4,h/2,w/2]
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x

    def fuse_channels(self, x, num):
        # [b,c,h,w]->[b,c/2,h,w]
        if num == 2:
            return (x[:, 0::2, ...] + x[:, 1::2, ...]) / 2
        elif num == 4:
            return (x[:, 0::4, ...] + x[:, 1::4, ...] + x[:, 2::4, ...] + x[:, 3::4, ...]) / 4
        elif num == 8:
            return (x[:, 0::8, ...] + x[:, 1::8, ...] + x[:, 2::8, ...] + x[:, 3::8, ...] + x[:, 4::8, ...] + x[:, 5::8, ...] + x[:, 6::8, ...] + x[:, 7::8, ...]) / 8
        else:
            raise NotImplementedError(f'num of {num} are not implemented !')

    def forward(self, x):  # l,m,s
        """
        #
        256, 512, 1024
        from small -> large
        """
        x_level_0 = x[3]  # max feature level [512,20,20]
        x_level_1 = x[2]  # mid feature level [256,40,40]
        x_level_2 = x[1]  # min feature level [128,80,80]
        x_level_3 = x[0]
        # print(x_level_0.shape)
        # print(x_level_1.shape)
        # print(x_level_2.shape)
        # print(x_level_3.shape)
        if self.type == 'ASFF':
            if self.level == 0:
                level_0_resized = x_level_0

                level_1_resized = self.stride_level_1(x_level_1)

                level_2_downsampled_inter = self.pool(x_level_2)
                level_2_resized = self.stride_level_2(
                    level_2_downsampled_inter)

                level_3_downsampled_inter = self.pool(x_level_3)
                level_3_downsampled_inter = self.pool(level_3_downsampled_inter)
                level_3_resized = self.stride_level_3(level_3_downsampled_inter)
            elif self.level == 1:
                level_0_compressed = self.compress_level_0(x_level_0)
                level_0_resized = self.upsample_level_0(level_0_compressed)

                level_1_resized = x_level_1

                level_2_resized = self.stride_level_2(x_level_2)

                level_3_downsampled_inter = self.pool(x_level_3)
                level_3_resized = self.stride_level_3(level_3_downsampled_inter)
            elif self.level == 2:
                level_0_compressed = self.compress_level_0(x_level_0)
                level_0_resized = self.upsample_level_0(level_0_compressed)


                level_1_compressed = self.compress_level_1(x_level_1)
                level_1_resized = self.upsample_level_1(level_1_compressed)

                level_2_resized = x_level_2

                level_3_resized = self.stride_level_3(x_level_3)
            elif self.level == 3:
                level_0_compressed = self.compress_level_0(x_level_0)
                level_0_resized = self.upsample_level_0(level_0_compressed)
  
                level_1_compressed = self.compress_level_1(x_level_1)
                level_1_resized = self.upsample_level_1(level_1_compressed)


                level_2_compressed = self.compress_level_2(x_level_2)
                level_2_resized = self.upsample_level_2(level_2_compressed)
 
                level_3_resized = x_level_3
        else:
            """
            x_level_0 = x[2]  # max feature level [512,20,20]
            x_level_1 = x[1]  # mid feature level [256,40,40]
            x_level_2 = x[0]  # min feature level [128,80,80]
            """
            # print("ASFFsim")
            if self.level == 0:
                level_0_resized = x_level_0

                level_1_resized = self.focus_channels(x_level_1)
                level_1_resized = self.fuse_channels(level_1_resized, 2)

                level_2_resized = self.focus_channels(x_level_2)
                level_2_resized = self.pool(level_2_resized)

                level_3_resized = self.focus_channels(x_level_3)
                level_3_resized = self.fuse_channels(level_3_resized, 2)
                level_3_resized = self.pool(level_3_resized)
                level_3_resized = self.focus_channels(level_3_resized)
            elif self.level == 1:
                level_0_resized = self.level_0_upsample(x_level_0)
                level_0_resized = self.fuse_channels(level_0_resized, 2)

                level_1_resized = x_level_1

                level_2_resized = self.focus_channels(x_level_2)
                level_2_resized = self.fuse_channels(level_2_resized, 2)

                level_3_resized = self.focus_channels(x_level_3)
                level_3_resized = self.pool(level_3_resized)
            elif self.level == 2:
                level_0_resized = self.level_0_upsample(x_level_0)
                level_0_resized = self.fuse_channels(level_0_resized, 4)


                level_1_resized = self.level_1_upsample(x_level_1)
                level_1_resized = self.fuse_channels(level_1_resized, 2)

                level_2_resized = x_level_2

                level_3_resized = self.focus_channels(x_level_3)
                level_3_resized = self.fuse_channels(level_3_resized, 2)
            elif self.level == 3:
                level_0_resized = self.level_0_upsample(x_level_0)
                level_0_resized = self.fuse_channels(level_0_resized, 8)

                level_1_resized = self.level_1_upsample(x_level_1)
                level_1_resized = self.fuse_channels(level_1_resized, 4)

                level_2_resized = self.level_2_upsample(x_level_2)
                level_2_resized = self.fuse_channels(level_2_resized, 2)

                level_3_resized = x_level_3

        level_0_weight_v = self.weight_level_0(level_0_resized)  # 3,20,20
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
            level_1_resized * levels_weight[:, 1:2, :, :] + \
            level_2_resized * levels_weight[:, 2:3, :, :] + \
            level_3_resized * levels_weight[:, 3:, :, :]
        out = self.expand(fused_out_reduced)

        return out

class TinyASFF3(nn.Module):

    def __init__(self,
                 level,
                 type='ASFF',
                 asff_channel=2,
                 expand_kernel=3,
                 multiplier=1,
                 use_carafe=False,
                 act='silu'):
        """
        Args:
            level(int): the level of the input feature
            type(str): ASFF or ASFF_sim
            asff_channel(int): the hidden channel of the attention layer in ASFF
            expand_kernel(int): expand kernel size of the expand layer
            multiplier: should be the same as width in the backbone
        """
        super(TinyASFF3, self).__init__()
        self.level = level
        self.type = type
        self.use_carafe = use_carafe
        
        self.dim = [
            int(1024 * multiplier), # 512
            int(512 * multiplier), # 256
            int(256 * multiplier) # 128
        ]

        Conv = BaseConv

        self.inter_dim = self.dim[self.level]

        if self.type == 'ASFF':
            if level == 0:
                self.stride_level_1 = Conv( # 256 * 512 /2
                    int(512 * multiplier), self.inter_dim, 3, 2, act=act)

                self.stride_level_2 = Conv( # 128 * 512 /2
                    int(256 * multiplier), self.inter_dim, 3, 2, act=act)

            elif level == 1:
                self.compress_level_0 = Conv( # 512 * 256 /1
                    int(1024 * multiplier), self.inter_dim, 1, 1, act=act)
                self.upsample_level_0 = CARAFEPack(channels=self.inter_dim, scale_factor=2) if self.use_carafe else nn.Upsample(scale_factor=2)
                self.stride_level_2 = Conv( # 128 * 256 /2
                    int(256 * multiplier), self.inter_dim, 3, 2, act=act)

            elif level == 2:
                self.compress_level_0 = Conv( # 512 * 128 /1
                    int(1024 * multiplier), self.inter_dim, 1, 1, act=act)
                self.upsample_level_0 = CARAFEPack(channels=self.inter_dim, scale_factor=2) if self.use_carafe else nn.Upsample(scale_factor=4)
                self.compress_level_1 = Conv( # 256 * 128 /1
                    int(512 * multiplier), self.inter_dim, 1, 1, act=act)
                self.upsample_level_1 = CARAFEPack(channels=self.inter_dim, scale_factor=2) if self.use_carafe else nn.Upsample(scale_factor=2)
            else:
                raise ValueError('Invalid level {}'.format(level))
        else:
            if level == 0:
                pass
            elif level == 1:
                self.level_0_upsample = CARAFEPack(channels=self.dim[0], scale_factor=2) if self.use_carafe else nn.Upsample(scale_factor=2)
            elif level == 2:
                self.level_0_upsample = CARAFEPack(channels=self.dim[0], scale_factor=4) if self.use_carafe else nn.Upsample(scale_factor=4)
                self.level_1_upsample = CARAFEPack(channels=self.dim[1], scale_factor=2) if self.use_carafe else nn.Upsample(scale_factor=2)
            else:
                raise ValueError('Invalid level {}'.format(level))
            
        # add expand layer
        self.expand = Conv( 
            self.inter_dim, self.inter_dim, expand_kernel, 1, act=act)

        self.weight_level_0 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)
        self.weight_level_1 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)
        self.weight_level_2 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)

        self.weight_levels = Conv(asff_channel * 3, 3, 1, 1, act=act)
        
        
    def init_weights(self):
        """Initialize the weights of module."""
        super(TinyASFF3, self).init_weights()
        for m in self.modules():
            if isinstance(m, CARAFEPack):
                m.init_weights()
                
                
    def expand_channel(self, x):
        # [b,c,h,w]->[b,c*4,h/2,w/2]
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x

    def mean_channel(self, x):
        # [b,c,h,w]->[b,c/4,h*2,w*2]
        x1 = x[:, ::2, :, :]
        x2 = x[:, 1::2, :, :]
        return (x1 + x2) / 2

    def forward(self, x):  # l,m,s
        """
        #
        256, 512, 1024
        from small -> large
        """
        x_level_0 = x[2]  # max feature level [512,20,20]
        x_level_1 = x[1]  # mid feature level [256,40,40]
        x_level_2 = x[0]  # min feature level [128,80,80]

        if self.type == 'ASFF':
            if self.level == 0:
                level_0_resized = x_level_0 # 512,20,20
                level_1_resized = self.stride_level_1(x_level_1) # 256,40,40 -> 256 * 512 / 2 -> 512,20,20
                level_2_downsampled_inter = F.max_pool2d( # 128,80,80 -> maxpool /2 -> 128,40,40
                    x_level_2, 3, stride=2, padding=1)
                level_2_resized = self.stride_level_2( # 128,40,40 -> 128 * 512 /2 -> 512,20,20
                    level_2_downsampled_inter)
            elif self.level == 1:
                level_0_compressed = self.compress_level_0(x_level_0) # 512,20,20 -> 512 * 256 /1 -> 256,20,20
                level_0_resized = self.upsample_level_0(level_0_compressed)
                level_1_resized = x_level_1 # 256,40,40
                level_2_resized = self.stride_level_2(x_level_2) # 128,80,80 -> 128 * 256 /2 -> 256,40,40
            elif self.level == 2:
                level_0_compressed = self.compress_level_0(x_level_0)
                level_0_resized = F.interpolate(
                    level_0_compressed, scale_factor=4, mode='nearest')
                level_1_compressed = self.compress_level_1(x_level_1)
                level_1_resized = self.upsample_level_1(level_1_compressed)
                level_2_resized = x_level_2
        else:
            if self.level == 0:
                level_0_resized = x_level_0
                level_1_resized = self.expand_channel(x_level_1)
                level_1_resized = self.mean_channel(level_1_resized)
                level_2_resized = self.expand_channel(x_level_2)
                level_2_resized = F.max_pool2d(
                    level_2_resized, 3, stride=2, padding=1)
            elif self.level == 1:
                level_0_resized = self.level_0_upsample(x_level_0)
                level_0_resized = self.mean_channel(level_0_resized)
                level_1_resized = x_level_1
                level_2_resized = self.expand_channel(x_level_2)
                level_2_resized = self.mean_channel(level_2_resized)

            elif self.level == 2:
                level_0_resized = self.level_0_upsample(x_level_0)
                level_0_resized = self.mean_channel(
                    self.mean_channel(level_0_resized))
                level_1_resized = self.level_1_upsample(x_level_1)
                level_1_resized = self.mean_channel(level_1_resized)
                level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:
                                                               1, :, :] + level_1_resized * levels_weight[:,
                                                                                            1:
                                                                                            2, :,
                                                                                            :] + level_2_resized * levels_weight[
                                                                                                                   :,
                                                                                                                   2:,
                                                                                                                   :, :]
        out = self.expand(fused_out_reduced)

        return out
    
    
@MODELS.register_module()
class TinyASFFNeck(nn.Module):
    def __init__(self, widen_factor, head_num=4, use_att='ASFF', groups=1, use_two_group_expand=False, use_group_expand_nums=1,use_carafe=False, use_softpool=False, asff_channel=2, expand_kernel=3, act='silu'):
        super().__init__()
        self.head_num = head_num
        if self.head_num == 4:
            self.asff_1 = ASFF(
                level=0,
                type=use_att,
                groups=groups,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=widen_factor,
                use_carafe=use_carafe,
                use_softpool=use_softpool,
                use_two_group_expand=use_two_group_expand,
                use_group_expand_nums=use_group_expand_nums,
                act=act,
            )
            self.asff_2 = ASFF(
                level=1,
                type=use_att,
                groups=groups,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=widen_factor,
                use_carafe=use_carafe,
                use_softpool=use_softpool,
                use_two_group_expand=use_two_group_expand,
                use_group_expand_nums=use_group_expand_nums,
                act=act,
            )
            self.asff_3 = ASFF(
                level=2,
                type=use_att,
                groups=groups,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=widen_factor,
                use_carafe=use_carafe,
                use_softpool=use_softpool,
                use_two_group_expand=use_two_group_expand,
                use_group_expand_nums=use_group_expand_nums,
                act=act,
            )
            self.asff_4 = ASFF(
                level=3,
                type=use_att,
                groups=groups,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=widen_factor,
                use_carafe=use_carafe,
                use_softpool=use_softpool,
                use_two_group_expand=use_two_group_expand,
                use_group_expand_nums=use_group_expand_nums,
                act=act,
            )
        elif self.head_num == 3:
            self.asff_1 = TinyASFF3(
                level=0,
                type=use_att,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=widen_factor,
                use_carafe=use_carafe,
                act=act,
            )
            self.asff_2 = TinyASFF3(
                level=1,
                type=use_att,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=widen_factor,
                use_carafe=use_carafe,
                act=act,
            )
            self.asff_3 = TinyASFF3(
                level=2,
                type=use_att,
                asff_channel=asff_channel,
                expand_kernel=expand_kernel,
                multiplier=widen_factor,
                use_carafe=use_carafe,
                act=act,
            )
        else:
                raise ValueError('Invalid head num {}'.format(self.head_num))
            
    def forward(self, x):
        if self.head_num == 4:
            pan_out0 = self.asff_1(x)
            pan_out1 = self.asff_2(x)
            pan_out2 = self.asff_3(x)
            pan_out3 = self.asff_4(x)
            outputs = (pan_out3, pan_out2, pan_out1, pan_out0)
        elif self.head_num == 3:
            pan_out0 = self.asff_1(x)
            pan_out1 = self.asff_2(x)
            pan_out2 = self.asff_3(x)
            outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

if __name__ == '__main__':
    input=[torch.randn(1,64,160,160), torch.randn(1,128,80,80), torch.randn(1,256,40,40), torch.randn(1,512,20,20)]
    model = TinyASFFNeck(use_att='TinyASFF', groups=4, use_two_group_expand=True, widen_factor=0.5, use_softpool=False, use_carafe=False)
    output = model(input)
    # print(output.shape)
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    from fvcore.nn import flop_count_str
    flops = FlopCountAnalysis(model, input)
    print(f'input shape: {input[0].shape}')
    print(flop_count_table(flops))
    # print(flop_count_str(flops))
    # input = torch.randn(1, 10, 41, 41)
    # out = F.max_pool2d(input, 3, stride=2, padding=1)
    # print(out.shape)
    