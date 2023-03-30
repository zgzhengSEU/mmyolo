_base_ = ['../../../configs/_base_/default_runtime.py', '../../../configs/_base_/det_p5_tta.py']
# ======================== wandb & run ==============================
TAGS = ["H100", "load", "p2","AdamW", 'CEPAFPN', 'SCA', 'TinyASFF']
GROUP_NAME = "yolov7_X"
ALGO_NAME = "yolov7_X_p2_AdamW_CEPAFPN_SCAg16-1234_TinyCEASFF"
DATASET_NAME = "VisDrone"

Wandb_init_kwargs = dict(
    project=DATASET_NAME,
    group=GROUP_NAME,
    name=ALGO_NAME,
    tags=TAGS,
    mode="online"
)
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=Wandb_init_kwargs)])

import datetime as dt
NOW_TIME = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = f"runs/{DATASET_NAME}/{ALGO_NAME}/{NOW_TIME}"

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_x_syncbn_fast_8x16b-300e_coco/yolov7_x_syncbn_fast_8x16b-300e_coco_20221124_215331-ef949a68.pth"

# ========================Frequently modified parameters======================
# -----data related-----
data_root = 'data/VisDrone/'  # Root path of data
CLASSES = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")
METAINFO = {'classes': CLASSES}
# Path of train annotation file
train_ann_file = 'annotations/train.json'
train_data_prefix = 'images/train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/val.json'
val_data_prefix = 'images/val/'  # Prefix of val image path

num_classes = 10  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 32
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8
# persistent_workers must be False if num_workers is 0
persistent_workers = True

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
# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
base_lr = 0.01
max_epochs = 300  # Maximum training epochs

num_epoch_stage2 = 30  # The last 30 epochs switch evaluation interval
val_interval_stage2 = 1  # Evaluation interval

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS.
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (640, 640)  # width, height
# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5)

# -----model related-----
strides = [4, 8, 16, 32]  # Strides of multi-scale prior box
num_det_layers = 4  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)

# Data augmentation
max_translate_ratio = 0.2  # YOLOv5RandomAffine
scaling_ratio_range = (0.1, 2.0)  # YOLOv5RandomAffine
mixup_prob = 0.15  # YOLOv5MixUp
randchoice_mosaic_prob = [0.8, 0.2]
mixup_alpha = 8.0  # YOLOv5MixUp
mixup_beta = 8.0  # YOLOv5MixUp

# -----train val related-----
loss_cls_weight = 0.3
loss_bbox_weight = 0.05
loss_obj_weight = 0.7
# BatchYOLOv7Assigner params
simota_candidate_topk = 10
simota_iou_weight = 3.0
simota_cls_weight = 1.0
prior_match_thr = 4.  # Priori box matching threshold
obj_level_weights = [4., 1.,
                     0.25, 0.06]  # The obj loss weights of the three output layers

lr_factor = 0.1  # Learning rate scaling factor
weight_decay = 0.0005
save_epoch_intervals = 1  # Save model checkpoint and validation intervals
max_keep_ckpts = 3  # The maximum checkpoints to keep.

# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(type='ShuffleCoordAttention', groups=16),
                stages=(True, True, True, True))
        ],
        out_indices=(1, 2, 3, 4),
        type='YOLOv7Backbone',
        arch='X',
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=[
        dict(
            use_carafe=True,
            type='YOLOv7PAFPN4',
            block_cfg=dict(
                type='ELANBlock',
                middle_ratio=0.4,
                block_ratio=0.4,
                num_blocks=3,
                num_convs_in_block=2),
            upsample_feats_cat_first=False,
            in_channels=[320, 640, 1280, 1280],
            # The real output channel will be multiplied by 2
            out_channels=[80, 160, 320, 640],
            use_repconv_outs=False,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True)),
        dict(
            type='TinyASFFNeck',
            widen_factor=1,
            use_carafe=True,
            use_att='ASFF-X')],
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            num_classes=num_classes,
            in_channels=[160, 320, 640, 1280],
            featmap_strides=strides,
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=anchors,
            strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_cls_weight *
            (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='mean',
            loss_weight=loss_bbox_weight * (3 / num_det_layers),
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_obj_weight *
            ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
        prior_match_thr=prior_match_thr,
        obj_level_weights=obj_level_weights,
        # BatchYOLOv7Assigner params
        simota_candidate_topk=simota_candidate_topk,
        simota_iou_weight=simota_iou_weight,
        simota_cls_weight=simota_cls_weight),
    test_cfg=model_test_cfg)

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True)
]

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
        max_translate_ratio=max_translate_ratio,  # note
        scaling_ratio_range=scaling_ratio_range,  # note
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
        max_translate_ratio=max_translate_ratio,  # note
        scaling_ratio_range=scaling_ratio_range,  # note
        # img_scale is (width, height)
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
        alpha=mixup_alpha,  # note
        beta=mixup_beta,  # note
        prob=mixup_prob,
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
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),  # FASTER
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=METAINFO,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=METAINFO,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='cosine',
        lr_factor=lr_factor,  # note
        max_epochs=max_epochs),
    checkpoint=dict(
        type='CheckpointHook',
        save_param_scheduler=False,
        interval=save_epoch_intervals,
        save_best='auto',
        max_keep_ckpts=max_keep_ckpts))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),  # Can be accelerated
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals,
    dynamic_intervals=[(max_epochs - num_epoch_stage2, val_interval_stage2)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
