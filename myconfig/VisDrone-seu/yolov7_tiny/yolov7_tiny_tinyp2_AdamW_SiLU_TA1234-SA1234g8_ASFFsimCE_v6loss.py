_base_ = './yolov7_l_origin.py'

# ======================== wandb & run ==============================
TAGS = ["SEU", "load", "tinyp2","AdamW", "SIOU", "ASFFsim", "TA", "SA", "SiLU", "v6loss"]
GROUP_NAME = "yolov7_tiny"
ALGO_NAME = "yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234g8_ASFFsimCE_v6loss"
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
train_batch_size_per_gpu = 8

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
        plugins=[
            dict(
                cfg=dict(type='TripletAttention'),
                stages=(True, True, True, True)),
            dict(
                cfg=dict(type='ShuffleAttention', groups=16),
                stages=(True, True, True, True))
        ],
        arch='Tiny', 
        # act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
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
    neck=[
        dict(
            type='YOLOv7PAFPN4',
            upsample_feats_cat_first=False,
            norm_cfg=norm_cfg,
            is_tiny_version=True,
            in_channels=[64, 128, 256, 512],
            out_channels=[32, 64, 128, 256], # 4 层时不会*2
            block_cfg=dict(
                type='TinyDownSampleBlock', middle_ratio=0.25),
            # act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
            act_cfg=dict(type='SiLU', inplace=True),
            use_repconv_outs=False),
        dict(
            type='ASFFNeck4',
            widen_factor=0.5,
            use_carafe=True,
            use_att='ASFFsim')],
    bbox_head=dict(
        _delete_=True,
        type='YOLOv6Head',
        head_module=dict(
            type='YOLOv7HeadModuleForYOLOv6',
            num_classes=num_classes,
            in_channels=[64, 128, 256, 512],
            featmap_strides=strides,
            num_base_priors=1),
        prior_generator = dict(
            type='mmdet.MlvlPointGenerator',
            offset=0.5,
            strides=strides),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='siou',
            bbox_format='xyxy',
            reduction='mean',
            loss_weight=2.5,
            return_iou=False)),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(
            type='BatchATSSAssigner',
            num_classes=num_classes,
            topk=9,
            iou_calculator=dict(type='mmdet.BboxOverlaps2D')),
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            topk=13,
            alpha=1,
            beta=6),
    ),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300))

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


"""