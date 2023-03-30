_base_ = './yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF.py'

# ======================== wandb & run ==============================
TAGS = ["SEU", "load", "tinyp2","AdamW", 'CEPAFPN', 'SCA', 'TinyASFF']
GROUP_NAME = "yolov7_tiny"
ALGO_NAME = "yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF_KD"
DATASET_NAME = "VisDrone"

Wandb_init_kwargs = dict(
    project=DATASET_NAME,
    group=GROUP_NAME,
    name=ALGO_NAME,
    tags=TAGS,
    mode="offline"
)
visualizer = dict(vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=Wandb_init_kwargs)])
# visualizer = dict(vis_backends = [dict(type='LocalVisBackend')])

# ====================================================================

teacher_ckpt = '/home/gp.sc.cc.tohoku.ac.jp/duanct/openmmlab/mmyolo/runs/VisDrone/yolov7_l_p2_AdamW_CEPAFPN_SCAg16-1234_TinyCEASFF/20230328_015114/best_coco/bbox_mAP_epoch_297.pth'  # noqa: E501

norm_cfg = dict(type='BN', affine=False, track_running_stats=False)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmyolo::VisDrone/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF.py'),
    teacher=dict(
        cfg_path='mmyolo::VisDrone/yolov7_tiny/yolov7_l_p2.py'),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        # `recorders` are used to record various intermediate results during
        # the model forward.
        student_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck.1.asff_4.expand.conv'),
            fpn1=dict(type='ModuleOutputs', source='neck.1.asff_3.expand.conv'),
            fpn2=dict(type='ModuleOutputs', source='neck.1.asff_2.expand.conv'),
            fpn3=dict(type='ModuleOutputs', source='neck.1.asff_1.expand.conv')),
        teacher_recorders=dict(
            fpn0=dict(type='ModuleOutputs', source='neck.1.asff_4.expand.conv'),
            fpn1=dict(type='ModuleOutputs', source='neck.1.asff_3.expand.conv'),
            fpn2=dict(type='ModuleOutputs', source='neck.1.asff_2.expand.conv'),
            fpn3=dict(type='ModuleOutputs', source='neck.1.asff_1.expand.conv')),
        # `connectors` are adaptive layers which usually map teacher's and
        # students features to the same dimension.
        connectors=dict(
                fpn0_s=dict(
                    type='ConvModuleConnector',
                    in_channel=64,
                    out_channel=128,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                fpn0_t=dict(
                    type='NormConnector', in_channels=128, norm_cfg=norm_cfg),
                fpn1_s=dict(
                    type='ConvModuleConnector',
                    in_channel=128,
                    out_channel=256,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                fpn1_t=dict(
                    type='NormConnector', in_channels=256, norm_cfg=norm_cfg),
                fpn2_s=dict(
                    type='ConvModuleConnector',
                    in_channel=256,
                    out_channel=512,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                fpn2_t=dict(
                    type='NormConnector', in_channels=512, norm_cfg=norm_cfg),
                fpn3_s=dict(
                    type='ConvModuleConnector',
                    in_channel=512,
                    out_channel=1024,
                    bias=False,
                    norm_cfg=norm_cfg,
                    act_cfg=None),
                fpn3_t=dict(
                    type='NormConnector', in_channels=1024, norm_cfg=norm_cfg)),
        distill_losses=dict(
            loss_fpn0=dict(type='ChannelWiseDivergence', loss_weight=1),
            loss_fpn1=dict(type='ChannelWiseDivergence', loss_weight=1),
            loss_fpn2=dict(type='ChannelWiseDivergence', loss_weight=1),
            loss_fpn3=dict(type='ChannelWiseDivergence', loss_weight=1)),
        # `loss_forward_mappings` are mappings between distill loss forward
        # arguments and records.
        loss_forward_mappings=dict(
            loss_fpn0=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn0', connector='fpn0_s'),
                preds_T=dict(
                    from_student=False, recorder='fpn0', connector='fpn0_t')),
            loss_fpn1=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn1', connector='fpn1_s'),
                preds_T=dict(
                    from_student=False, recorder='fpn1', connector='fpn1_t')),
            loss_fpn2=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn2', connector='fpn2_s'),
                preds_T=dict(
                    from_student=False, recorder='fpn2', connector='fpn2_t')),
            loss_fpn3=dict(
                preds_S=dict(
                    from_student=True, recorder='fpn3', connector='fpn3_s'),
                preds_T=dict(
                    from_student=False, recorder='fpn3', connector='fpn3_t')))))

find_unused_parameters = True

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
