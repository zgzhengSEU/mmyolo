_base_ = './base_config/yolov5_s-v61_syncbn_8xb16-300e_coco.py'

# ======================== wandb & run ==============================
TAGS = ["SEU", "load", "yolov5_s"]
GROUP_NAME = "Baseline_visdronepretrain"
ALGO_NAME = "yolov5_s_v61"
DATASET_NAME = "BJHDrone"

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

load_from = "https://download.openmmlab.com/mmyolo/v0/yolov5/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/yolov5_s-v61_syncbn_fast_8xb16-300e_coco_20220918_084700-86e02187.pth"
# ========================modified parameters========================


# fast means faster training speed,
# but less flexibility for multitasking
model = dict(
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True))

train_dataloader = dict(collate_fn=dict(type='yolov5_collate'))
