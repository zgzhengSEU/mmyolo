_base_ = './yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF.py'

# ======================== wandb & run ==============================
TAGS = ["SEU", "load", "tinyp2","AdamW", 'CEPAFPN', 'SCA', 'TinyASFF']
GROUP_NAME = "yolov7_tiny"
ALGO_NAME = "yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF_MGD"
DATASET_NAME = "VisDrone"

Wandb_init_kwargs = dict(
    project=DATASET_NAME,
    group=GROUP_NAME,
    name=ALGO_NAME,
    tags=TAGS,
    # resume='allow',
    # id='ct8z62cl',
    # allow_val_change=True,
    mode="online"
)
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=Wandb_init_kwargs)])
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend')])
import datetime as dt
NOW_TIME = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
work_dir = f"runs/{DATASET_NAME}/{ALGO_NAME}/{NOW_TIME}"
# ====================================================================

teacher_ckpt = '/home/gp.sc.cc.tohoku.ac.jp/duanct/openmmlab/mmyolo/runs/VisDrone/yolov7_X_p2_AdamW_CEPAFPN_SCAg16-1234_TinyCEASFF/20230331_120755/best_coco/bbox_mAP_epoch_284.pth'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    _delete_=True,
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmyolo::VisDrone/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF.py'),
    teacher=dict(
        cfg_path='mmyolo::VisDrone/yolov7_tiny/yolov7_X_p2_AdamW_CEPAFPN_SCAg16-1234_TinyCEASFF.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck.0.out_layers.0.conv'),
            fpn1=dict(type='ModuleOutputs', source='neck.0.out_layers.1.conv'),
            fpn2=dict(type='ModuleOutputs', source='neck.0.out_layers.2.conv'),
            fpn3=dict(type='ModuleOutputs', source='neck.0.out_layers.3.conv')),
        teacher_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck.0.out_layers.0.conv'),
            fpn1=dict(type='ModuleOutputs', source='neck.0.out_layers.1.conv'),
            fpn2=dict(type='ModuleOutputs', source='neck.0.out_layers.2.conv'),
            fpn3=dict(type='ModuleOutputs', source='neck.0.out_layers.3.conv')),
        connectors=dict(
            s_fpn0_connector=dict(
                type='MGDConnector',
                student_channels=64,
                teacher_channels=160,
                lambda_mgd=0.65),
            s_fpn1_connector=dict(
                type='MGDConnector',
                student_channels=128,
                teacher_channels=320,
                lambda_mgd=0.65),
            s_fpn2_connector=dict(
                type='MGDConnector',
                student_channels=256,
                teacher_channels=640,
                lambda_mgd=0.65),
            s_fpn3_connector=dict(
                type='MGDConnector',
                student_channels=512,
                teacher_channels=1280,
                lambda_mgd=0.65)),
        distill_losses=dict(
            loss_mgd_fpn0=dict(type='MGDLoss', alpha_mgd=0.00002),
            loss_mgd_fpn1=dict(type='MGDLoss', alpha_mgd=0.00002),
            loss_mgd_fpn2=dict(type='MGDLoss', alpha_mgd=0.00002),
            loss_mgd_fpn3=dict(type='MGDLoss', alpha_mgd=0.00002)),
        loss_forward_mappings=dict(
            loss_mgd_fpn0=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='fpn0',
                    connector='s_fpn0_connector'),
                preds_T=dict(from_student=False, recorder='fpn0')),
            loss_mgd_fpn1=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='fpn1',
                    connector='s_fpn1_connector'),
                preds_T=dict(from_student=False, recorder='fpn1')),
            loss_mgd_fpn2=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='fpn2',
                    connector='s_fpn2_connector'),
                preds_T=dict(from_student=False, recorder='fpn2')),
            loss_mgd_fpn3=dict(
                preds_S=dict(
                    from_student=True,
                    recorder='fpn3',
                    connector='s_fpn3_connector'),
                preds_T=dict(from_student=False, recorder='fpn3')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# =============================


# learning rate
base_lr=0.004
lr_start_factor = 1.0e-5
max_epochs=300
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=lr_start_factor,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

default_hooks = dict(
    _delete_=True,
    checkpoint=dict(
        type='CheckpointHook',
        save_param_scheduler=False,
        interval=1,
        save_best='auto',
        max_keep_ckpts=3))

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=(640, 640),
        ratio_range=(0.5, 2.0),
        resize_type='mmdet.Resize',
        keep_ratio=True),
    dict(type='mmdet.RandomCrop', crop_size=(640, 640)),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=300 - 20,
        switch_pipeline=train_pipeline_stage2),
    # stop distillation after the 280th epoch
    dict(type='mmrazor.StopDistillHook', stop_epoch=280)
]
