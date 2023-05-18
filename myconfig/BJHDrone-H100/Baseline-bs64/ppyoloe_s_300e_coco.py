_base_ = ['../../../configs/_base_/default_runtime.py', '../../../configs/_base_/det_p5_tta.py']

# ======================== wandb & run ==============================
TAGS = ["SEU", "load", "ppyoloe_s"]
GROUP_NAME = "Baseline_visdronepretrain"
ALGO_NAME = "ppyoloe_s_300e_coco"
DATASET_NAME = "BJHDrone"

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

# load_from = "https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco/ppyoloe_plus_s_fast_8xb8-80e_coco_20230101_154052-9fee7619.pth"
# ========================modified parameters========================

# dataset settings
data_root = 'data/VisDrone/'  # Root path of data
dataset_type = 'YOLOv5CocoDataset'
CLASSES = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")
METAINFO = {'classes': CLASSES}
# Path of train annotation file
train_ann_file = 'annotations/train.json'
train_data_prefix = 'images/train/'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'annotations/val.json'
val_data_prefix = 'images/val/'  # Prefix of val image path
# Path of test annotatino file
test_ann_file = 'annotations/test.json'
test_data_prefix = 'images/test/'  # Prefix of val image path

num_classes = 10  # Number of classes for classification
# Batch size of a single GPU during training
train_batch_size_per_gpu = 64
# Worker to pre-fetch data for each single GPU during training
train_num_workers = 8


# parameters that often need to be modified
img_scale = (640, 640)  # width, height
deepen_factor = 0.33
widen_factor = 0.5
max_epochs = 300

save_epoch_intervals = 1

val_batch_size_per_gpu = 1
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 2

# The pretrained model is geted and converted from official PPYOLOE.
# https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/configs/ppyoloe/README.md
checkpoint = 'https://download.openmmlab.com/mmyolo/v0/ppyoloe/ppyoloe_pretrain/cspresnet_s_imagenet1k_pretrained-2be81763.pth'  # noqa

# persistent_workers must be False if num_workers is 0.
persistent_workers = True

# Base learning rate for optim_wrapper
base_lr = 0.01

strides = [8, 16, 32]

model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        # use this to support multi_scale training
        type='PPYOLOEDetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='PPYOLOEBatchRandomResize',
                random_size_range=(320, 800),
                interval=1,
                size_divisor=32,
                random_interp=True,
                keep_ratio=False)
        ],
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255., 0.224 * 255., 0.225 * 255.],
        bgr_to_rgb=True),
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint=checkpoint,
            map_location='cpu'),
        type='PPYOLOECSPResNet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        block_cfg=dict(
            type='PPYOLOEBasicBlock', shortcut=True, use_alpha=False),
        norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
        act_cfg=dict(type='SiLU', inplace=True),
        attention_cfg=dict(
            type='EffectiveSELayer', act_cfg=dict(type='HSigmoid')),
        use_large_stem=True),
    neck=dict(
        type='PPYOLOECSPPAFPN',
        in_channels=[256, 512, 1024],
        out_channels=[192, 384, 768],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        num_csplayer=1,
        num_blocks_per_layer=3,
        block_cfg=dict(
            type='PPYOLOEBasicBlock', shortcut=False, use_alpha=False),
        norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
        act_cfg=dict(type='SiLU', inplace=True),
        drop_block_cfg=None,
        use_spp=True),
    bbox_head=dict(
        type='PPYOLOEHead',
        head_module=dict(
            type='PPYOLOEHeadModule',
            num_classes=num_classes,
            in_channels=[192, 384, 768],
            widen_factor=widen_factor,
            featmap_strides=strides,
            reg_max=16,
            norm_cfg=dict(type='BN', momentum=0.1, eps=1e-5),
            act_cfg=dict(type='SiLU', inplace=True),
            num_base_priors=1),
        prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='giou',
            bbox_format='xyxy',
            reduction='mean',
            loss_weight=2.5,
            return_iou=False),
        # Since the dflloss is implemented differently in the official
        # and mmdet, we're going to divide loss_weight by 4.
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=0.5 / 4)),
    train_cfg=dict(
        initial_epoch=100,
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
            beta=6,
            eps=1e-9)),
    test_cfg=dict(
        multi_label=True,
        nms_pre=1000,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.7),
        max_per_img=300))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PPYOLOERandomDistort'),
    dict(type='mmdet.Expand', mean=(103.53, 116.28, 123.675)),
    dict(type='PPYOLOERandomCrop'),
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
    collate_fn=dict(type='yolov5_collate', use_ms_training=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=METAINFO,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(
        type='mmdet.FixShapeResize',
        width=img_scale[0],
        height=img_scale[1],
        keep_ratio=False,
        interpolation='bicubic'),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
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
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        ann_file=val_ann_file,
        pipeline=test_pipeline))

test_dataloader = dict(
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
        data_prefix=dict(img=test_data_prefix),
        filter_cfg=dict(filter_empty_gt=True, min_size=0),
        ann_file=test_ann_file,
        pipeline=test_pipeline))

param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=False),
    paramwise_cfg=dict(norm_decay_mult=0.))

default_hooks = dict(
    param_scheduler=dict(
        type='PPYOLOEParamSchedulerHook',
        warmup_min_iter=1000,
        start_factor=0.,
        warmup_epochs=5,
        min_lr_ratio=0.0,
        total_epochs=int(max_epochs * 1.2)),
    checkpoint=dict(
        type='CheckpointHook',
        interval=save_epoch_intervals,
        save_best='auto',
        max_keep_ckpts=3))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + val_ann_file,
    metric='bbox')
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root + test_ann_file,
    metric='bbox')

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
