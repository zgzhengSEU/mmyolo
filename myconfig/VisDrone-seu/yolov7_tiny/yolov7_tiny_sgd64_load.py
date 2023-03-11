_base_ = './yolov7_l_origin.py'

# ======================== wandb & run ==============================
TAGS = ["SEU", "load", "yolov7_tiny", "sgd64"]
GROUP_NAME = "yolov7_tiny"
ALGO_NAME = "yolov7_tiny_sgd64_load"
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

# -----model related-----
# Basic size of multi-scale prior box
VisDrone_anchors_v5_k_means = [
    [(3, 5), (4, 9), (8, 6)], 
    [(8, 14), (16, 9), (15, 18)], 
    [(31, 17), (22, 35), (53, 38)]
]
DE_anchors = [
    [(3, 4), (4, 8), (7, 6)], 
    [(7, 12), (14, 9), (11, 18)], 
    [(25, 14), (21, 27), (44, 35)]
]
origin_anchors = [
    [(12, 16), (19, 36), (40, 28)],  # P3/8
    [(36, 75), (76, 55), (72, 146)],  # P4/16
    [(142, 110), (192, 243), (459, 401)]  # P5/32
]
anchors = VisDrone_anchors_v5_k_means # 修改anchor

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
num_det_layers = _base_.num_det_layers
img_scale = _base_.img_scale
pre_transform = _base_.pre_transform
model = dict(
    backbone=dict(
        arch='Tiny', act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    neck=dict(
        is_tiny_version=True,
        in_channels=[128, 256, 512],
        out_channels=[64, 128, 256],
        block_cfg=dict(
            _delete_=True, type='TinyDownSampleBlock', middle_ratio=0.25),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        use_repconv_outs=False),
    bbox_head=dict(
        head_module=dict(in_channels=[128, 256, 512]),
        prior_generator=dict(base_sizes=anchors),
        loss_cls=dict(loss_weight=loss_cls_weight *
                      (num_classes / 80 * 3 / num_det_layers)),
        loss_obj=dict(loss_weight=loss_obj_weight *
                      ((img_scale[0] / 640)**2 * 3 / num_det_layers))))

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
weight_decay = _base_.weight_decay

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv7OptimWrapperConstructor')

default_hooks = dict(param_scheduler=dict(lr_factor=lr_factor))
