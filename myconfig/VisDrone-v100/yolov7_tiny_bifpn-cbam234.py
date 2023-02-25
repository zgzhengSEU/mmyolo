_base_ = './yolov7_l_syncbn_fast_8x16b-300e_coco.py'

# =======================custom set==========================================
TAGS = ["v100", "yolov7_tiny", "300e"]
GROUP_NAME = "yolov7_tiny"
ALGO_NAME = "yolov7_tiny"
DATASET_NAME = "VisDrone"

CLASSES = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")
METAINFO = {'classes': CLASSES}

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov7/yolov7_tiny_syncbn_fast_8x16b-300e_coco/yolov7_tiny_syncbn_fast_8x16b-300e_coco_20221126_102719-0ee5bbdf.pth"

Wandb_init_kwargs = dict(
    project=DATASET_NAME,
    group=GROUP_NAME,
    name=ALGO_NAME,
    tags=TAGS
)
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=Wandb_init_kwargs)])
import datetime as dt
NOW_TIME = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = f"runs/{DATASET_NAME}/{ALGO_NAME}/{NOW_TIME}"

# ========================modified parameters========================

# -----model related-----
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
        arch='Tiny', act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        plugins=[
            dict(
                cfg=dict(type='CBAM'),
                stages=(False, False, True, True))
        ]),
    neck=dict(
        _delete_ = True,
        num_stages=3,
        in_channels=[128, 256, 512],
        out_channels=[64, 128, 256],
        start_level=0,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001)
        ),
    bbox_head=dict(
        head_module=dict(in_channels=[128, 256, 512]),
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

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
default_hooks = dict(param_scheduler=dict(lr_factor=lr_factor))

# SGD -> AdamW
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.005, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))