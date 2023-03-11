
bsub -J yolov7_tiny_AdamW -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_AdamW.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TA1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_TA1234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TA234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_TA234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TA34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_TA34.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SIOU.py --amp"


bsub -J yolov7_tiny_tinyp2_AdamW_ASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_ASFF.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_ASFFCE -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_ASFFCE.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_ASFFsim -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_ASFFsim.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_ASFFsimCE -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_ASFFsimCE.py --amp"
