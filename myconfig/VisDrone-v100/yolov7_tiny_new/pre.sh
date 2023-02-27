



CUDA_VISIBLE_DEVICES=0 python tools/train.py myconfig/VisDrone-v100/yolov7_tiny_new/yolov7_tiny_p2_autoAdamW.py --amp
CUDA_VISIBLE_DEVICES=1 python tools/train.py myconfig/VisDrone-v100/yolov7_tiny_new/yolov7_tiny_p2_autoSGD.py --amp
CUDA_VISIBLE_DEVICES=2 python tools/train.py myconfig/VisDrone-v100/yolov7_tiny_new/yolov7_tiny_p2_originAdamW.py --amp
CUDA_VISIBLE_DEVICES=3 python tools/train.py myconfig/VisDrone-v100/yolov7_tiny_new/yolov7_tiny_p2_originSGD.py --amp