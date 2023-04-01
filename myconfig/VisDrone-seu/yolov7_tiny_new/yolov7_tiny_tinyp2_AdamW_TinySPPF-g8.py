_base_ = './yolov7_l_origin.py'

# ======================== wandb & run ==============================
TAGS = ["SEU", "load", "tinyp2","AdamW", "TinySPPF"]
GROUP_NAME = "yolov7_tiny"
ALGO_NAME = "yolov7_tiny_tinyp2_AdamW_TinySPPF-g8"
DATASET_NAME = "VisDrone"

Wandb_init_kwargs = dict(
    project=DATASET_NAME,
    group=GROUP_NAME,
    name=ALGO_NAME,
    tags=TAGS,
    mode="offline"
)
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=Wandb_init_kwargs)])

import datetime as dt
NOW_TIME = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = f"runs/{DATASET_NAME}/{ALGO_NAME}/{NOW_TIME}"

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco/yolov7_tiny_syncbn_fast_8x16b-300e_coco_20221126_102719-0ee5bbdf.pth"
# ========================modified parameters========================
num_det_layers = 4
loss_bbox_weight = 0.05
strides = [4, 8, 16, 32]  # Strides of multi-scale prior box
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)
obj_level_weights=[4.0, 1.0, 0.25, 0.06]
# -----model related-----
# Basic size of multi-scale prior box
v5_k_means = [
    [[3, 4], [3, 7], [6, 6]], 
    [[6, 11], [14, 7], [10, 15]], 
    [[17, 12], [17, 23], [31, 16]], 
    [[27, 36], [54, 28], [67, 82]]
]
k_means = [
    [[4, 5], [6, 10], [10, 7]], 
    [[10, 17], [18, 11], [16, 25]], 
    [[29, 16], [26, 37], [45, 26]], 
    [[42, 59], [71, 41], [95, 86]]
]
DE = [
    [[3, 4], [4, 7], [6, 5]], 
    [[5, 11], [10, 7], [9, 15]], 
    [[16, 10], [14, 21], [24, 14]], 
    [[23, 31], [37, 21], [51, 44]]
]
anchors = v5_k_means # 修改anchor

# ---- data related -------
train_batch_size_per_gpu = 64

# Data augmentation
max_translate_ratio = 0.1  # YOLOv5RandomAffine
scaling_ratio_range = (0.5, 1.6)  # YOLOv5RandomAffine
mixup_prob = 0.05  # YOLOv5MixUp
randchoice_mosaic_prob = [0.8, 0.2]
mixup_alpha = 8.0  # YOLOv5MixUp
mixup_beta = 8.0  # YOLOv5MixUp

# -----train val related-----
loss_cls_weight = 0.5
loss_obj_weight = 1.0

lr_factor = 0.01  # Learning rate scaling factor
# ===============================Unmodified in most cases====================
num_classes = _base_.num_classes
img_scale = _base_.img_scale
pre_transform = _base_.pre_transform
model = dict(
    backbone=dict(
        arch='Tiny', 
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        out_indices=(1, 2, 3, 4)),
    # backbone=dict(
    #     arch='Tiny', 
    #     act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    #     out_indices=(1, 2, 3, 4),
    #     plugins=[
    #         dict(
    #             cfg=dict(type='CBAM'),
    #             stages=(True, True, True, True))
    #     ]),    
    neck=dict(
        type='YOLOv7PAFPN4',
        use_carafe=False,
        is_tiny_version=True,
        in_channels=[64, 128, 256, 512],
        out_channels=[32, 64, 128, 256],
        sppf_groups=8,
        block_cfg=dict(
            _delete_=True, type='TinyDownSampleBlock', middle_ratio=0.25),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        use_repconv_outs=False),
    bbox_head=dict(
        head_module=dict(
            in_channels = [64, 128, 256, 512],
            featmap_strides=strides),
        prior_generator=dict(base_sizes=anchors, strides=strides),
        obj_level_weights=obj_level_weights,
        loss_cls=dict(loss_weight=loss_cls_weight * (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(loss_weight=loss_bbox_weight * (3 / num_det_layers)),
        loss_obj=dict(loss_weight=loss_obj_weight * ((img_scale[0] / 640)**2 * 3 / num_det_layers))))

mosiac4_pipeline = [
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,  # change
        scaling_ratio_range=scaling_ratio_range,  # change
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
]

mosiac9_pipeline = [
    dict(
        type='Mosaic9',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,  # change
        scaling_ratio_range=scaling_ratio_range,  # change
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
]

randchoice_mosaic_pipeline = dict(
    type='RandomChoice',
    transforms=[mosiac4_pipeline, mosiac9_pipeline],
    prob=randchoice_mosaic_prob)

train_pipeline = [
    *pre_transform,
    randchoice_mosaic_pipeline,
    dict(
        type='YOLOv5MixUp',
        alpha=mixup_alpha,
        beta=mixup_beta,
        prob=mixup_prob,  # change
        pre_transform=[*pre_transform, randchoice_mosaic_pipeline]),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(pipeline=train_pipeline))

base_lr = (train_batch_size_per_gpu / 128) * _base_.base_lr
# base_lr = _base_.base_lr
weight_decay = _base_.weight_decay

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='SGD',
#         lr=base_lr,
#         momentum=0.937,
#         weight_decay=weight_decay,
#         nesterov=True,
#         batch_size_per_gpu=train_batch_size_per_gpu),
#     constructor='YOLOv7OptimWrapperConstructor')

# SGD -> AdamW
base_lr = 0.004
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))


default_hooks = dict(param_scheduler=dict(lr_factor=lr_factor))


"""

+-----------------------------------------+----------------------+------------+--------------+
| module                                  | #parameters or shape | #flops     | #activations |
+-----------------------------------------+----------------------+------------+--------------+
| model                                   | 6.128M               | 7.779G     | 28.103M      |
|  backbone                               |  2.464M              |  3.577G    |  14.131M     |
|   backbone.stem                         |   19.488K            |   0.57G    |   4.915M     |
|    backbone.stem.0                      |    0.928K            |    95.027M |    3.277M    |
|    backbone.stem.1                      |    18.56K            |    0.475G  |    1.638M    |
|   backbone.stage1.0                     |   31.104K            |   0.796G   |   4.915M     |
|    backbone.stage1.0.short_conv         |    2.112K            |    54.067M |    0.819M    |
|    backbone.stage1.0.main_convs         |    20.672K           |    0.529G  |    2.458M    |
|    backbone.stage1.0.final_conv         |    8.32K             |    0.213G  |    1.638M    |
|   backbone.stage2.1                     |   0.115M             |   0.739G   |   2.458M     |
|    backbone.stage2.1.short_conv         |    4.224K            |    27.034M |    0.41M     |
|    backbone.stage2.1.main_convs         |    78.208K           |    0.501G  |    1.229M    |
|    backbone.stage2.1.final_conv         |    33.024K           |    0.211G  |    0.819M    |
|   backbone.stage3.1                     |   0.46M              |   0.736G   |   1.229M     |
|    backbone.stage3.1.short_conv         |    16.64K            |    26.624M |    0.205M    |
|    backbone.stage3.1.main_convs         |    0.312M            |    0.499G  |    0.614M    |
|    backbone.stage3.1.final_conv         |    0.132M            |    0.211G  |    0.41M     |
|   backbone.stage4.1                     |   1.838M             |   0.735G   |   0.614M     |
|    backbone.stage4.1.short_conv         |    66.048K           |    26.419M |    0.102M    |
|    backbone.stage4.1.main_convs         |    1.247M            |    0.499G  |    0.307M    |
|    backbone.stage4.1.final_conv         |    0.525M            |    0.21G   |    0.205M    |
|  neck                                   |  3.619M              |  4.063G    |  12.442M     |
|   neck.reduce_layers                    |   0.701M             |   0.423G   |   1.843M     |
|    neck.reduce_layers.0                 |    2.112K            |    54.067M |    0.819M    |
|    neck.reduce_layers.1                 |    8.32K             |    53.248M |    0.41M     |
|    neck.reduce_layers.2                 |    33.024K           |    52.838M |    0.205M    |
|    neck.reduce_layers.3                 |    0.657M            |    0.263G  |    0.41M     |
|   neck.upsample_layers                  |   43.456K            |   41.472M  |   0.358M     |
|    neck.upsample_layers.0               |    33.024K           |    13.414M |    51.2K     |
|    neck.upsample_layers.1               |    8.32K             |    13.722M |    0.102M    |
|    neck.upsample_layers.2               |    2.112K            |    14.336M |    0.205M    |
|   neck.top_down_layers                  |   0.184M             |   0.677G   |   4.301M     |
|    neck.top_down_layers.0               |    0.14M             |    0.224G  |    0.614M    |
|    neck.top_down_layers.1               |    35.2K             |    0.225G  |    1.229M    |
|    neck.top_down_layers.2               |    8.896K            |    0.228G  |    2.458M    |
|   neck.downsample_layers                |   0.388M             |   0.355G   |   0.717M     |
|    neck.downsample_layers.0             |    18.56K            |    0.119G  |    0.41M     |
|    neck.downsample_layers.1             |    73.984K           |    0.118G  |    0.205M    |
|    neck.downsample_layers.2             |    0.295M            |    0.118G  |    0.102M    |
|   neck.bottom_up_layers                 |   0.734M             |   0.673G   |   2.15M      |
|    neck.bottom_up_layers.0              |    35.2K             |    0.225G  |    1.229M    |
|    neck.bottom_up_layers.1              |    0.14M             |    0.224G  |    0.614M    |
|    neck.bottom_up_layers.2              |    0.559M            |    0.223G  |    0.307M    |
|   neck.out_layers                       |   1.569M             |   1.894G   |   3.072M     |
|    neck.out_layers.0                    |    18.56K            |    0.475G  |    1.638M    |
|    neck.out_layers.1                    |    73.984K           |    0.473G  |    0.819M    |
|    neck.out_layers.2                    |    0.295M            |    0.473G  |    0.41M     |
|    neck.out_layers.3                    |    1.181M            |    0.472G  |    0.205M    |
|  bbox_head.head_module.convs_pred       |  44.52K              |  0.138G    |  1.53M       |
|   bbox_head.head_module.convs_pred.0    |   3.034K             |   73.728M  |   1.152M     |
|    bbox_head.head_module.convs_pred.0.0 |    64                |    0       |    0         |
|    bbox_head.head_module.convs_pred.0.1 |    2.925K            |    73.728M |    1.152M    |
|    bbox_head.head_module.convs_pred.0.2 |    45                |    0       |    0         |
|   bbox_head.head_module.convs_pred.1    |   5.978K             |   36.864M  |   0.288M     |
|    bbox_head.head_module.convs_pred.1.0 |    0.128K            |    0       |    0         |
|    bbox_head.head_module.convs_pred.1.1 |    5.805K            |    36.864M |    0.288M    |
|    bbox_head.head_module.convs_pred.1.2 |    45                |    0       |    0         |
|   bbox_head.head_module.convs_pred.2    |   11.866K            |   18.432M  |   72K        |
|    bbox_head.head_module.convs_pred.2.0 |    0.256K            |    0       |    0         |
|    bbox_head.head_module.convs_pred.2.1 |    11.565K           |    18.432M |    72K       |
|    bbox_head.head_module.convs_pred.2.2 |    45                |    0       |    0         |
|   bbox_head.head_module.convs_pred.3    |   23.642K            |   9.216M   |   18K        |
|    bbox_head.head_module.convs_pred.3.0 |    0.512K            |    0       |    0         |
|    bbox_head.head_module.convs_pred.3.1 |    23.085K           |    9.216M  |    18K       |
|    bbox_head.head_module.convs_pred.3.2 |    45                |    0       |    0         |
+-----------------------------------------+----------------------+------------+--------------+


==============================
Input shape: torch.Size([640, 640])
Model Flops: 7.779G
Model Parameters: 6.128M
==============================
"""