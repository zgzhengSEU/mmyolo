_base_ = ['./yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF.py']

# ======================== wandb & run ==============================
TAGS = ["SEU", "load", "tinyp2","AdamW", 'CEPAFPN', 'SCA', 'TinyASFF']
GROUP_NAME = "yolov7_tiny"
ALGO_NAME = "yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF_fromX_FBKD"
DATASET_NAME = "VisDrone"

Wandb_init_kwargs = dict(
    project=DATASET_NAME,
    group=GROUP_NAME,
    name=ALGO_NAME,
    tags=TAGS,
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
    _delete_=True,
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=dict(
        cfg_path='mmyolo::VisDrone/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF.py'),
    teacher=dict(
        cfg_path='mmyolo::VisDrone/yolov7_tiny/yolov7_X_p2_AdamW_CEPAFPN_SCAg16-1234_TinyCEASFF.py'),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            neck_s0=dict(type='ModuleOutputs', source='neck.0.out_layers.0.conv'),
            neck_s1=dict(type='ModuleOutputs', source='neck.0.out_layers.1.conv'),
            neck_s2=dict(type='ModuleOutputs', source='neck.0.out_layers.2.conv'),
            neck_s3=dict(type='ModuleOutputs', source='neck.0.out_layers.3.conv')),
        teacher_recorders=dict(
            neck_s0=dict(type='ModuleOutputs', source='neck.0.out_layers.0.conv'),
            neck_s1=dict(type='ModuleOutputs', source='neck.0.out_layers.1.conv'),
            neck_s2=dict(type='ModuleOutputs', source='neck.0.out_layers.2.conv'),
            neck_s3=dict(type='ModuleOutputs', source='neck.0.out_layers.3.conv')),
        distill_losses=dict(
            loss_s0=dict(type='FBKDLoss'),
            loss_s1=dict(type='FBKDLoss'),
            loss_s2=dict(type='FBKDLoss'),
            loss_s3=dict(type='FBKDLoss')),
        connectors=dict(
            loss_s0_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels_conv=64,
                out_channels=160,
                in_channels=160,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=8),
            loss_s0_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=160,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=8),
            loss_s1_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels_conv=128,
                out_channels=320,
                in_channels=320,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=4),
            loss_s1_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=320,
                reduction=4,
                mode='dot_product',
                sub_sample=True,
                maxpool_stride=4),
            loss_s2_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels_conv=256,
                in_channels=640,
                out_channels=640,
                mode='dot_product',
                sub_sample=True),
            loss_s2_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=640,
                mode='dot_product',
                sub_sample=True),
            loss_s3_sfeat=dict(
                type='FBKDStudentConnector',
                in_channels_conv=512,
                out_channels=1280,
                in_channels=1280,
                mode='dot_product',
                sub_sample=True),
            loss_s3_tfeat=dict(
                type='FBKDTeacherConnector',
                in_channels=1280,
                mode='dot_product',
                sub_sample=True)),
        loss_forward_mappings=dict(
            loss_s0=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s0',
                    connector='loss_s0_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s0',
                    connector='loss_s0_tfeat')),
            loss_s1=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s1',
                    connector='loss_s1_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s1',
                    connector='loss_s1_tfeat')),
            loss_s2=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s2',
                    connector='loss_s2_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s2',
                    connector='loss_s2_tfeat')),
            loss_s3=dict(
                s_input=dict(
                    from_student=True,
                    recorder='neck_s3',
                    connector='loss_s3_sfeat'),
                t_input=dict(
                    from_student=False,
                    recorder='neck_s3',
                    connector='loss_s3_tfeat')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

# =============================

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', file_client_args=_base_.file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='mmdet.RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
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
        switch_epoch=300 - 30,
        switch_pipeline=train_pipeline_stage2),
    # stop distillation after the 280th epoch
    dict(type='mmrazor.StopDistillHook', stop_epoch=280)
]

optim_wrapper = dict(clip_grad=dict(max_norm=35))
