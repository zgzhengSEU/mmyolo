bsub -J yolov7_tiny_sgd64 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_sgd64.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_TA1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_TA1234.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_TA234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_TA234.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_TA34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_TA34.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_SIOU.py --amp"


bsub -J yolov7_tiny_tinyp2_sgd64_ASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_ASFF.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_ASFFCE -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_ASFFCE.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_ASFFsim -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_ASFFsim.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_ASFFsimCE -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_ASFFsimCE.py --amp"

bsub -J yolov7_tiny_sgd64_load -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_sgd64_load.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_v6loss -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_v6loss.py --amp"


# ======================================================
bsub -J yolov7_tiny_tinyp2_AdamW_TA4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_TA4.py --amp"






# ===============================================

bsub -J yolov7_tiny_tinyp2_AdamW_CA4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA34.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA12344 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA12344.py --amp"

# =========================

bsub -J yolov7_tiny_tinyp2_AdamW_SA4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SA4.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SA34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SA34.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SA234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SA234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SA12344 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SA12344.py --amp"

# ==========================



bsub -J yolov7_tiny_tinyp2_AdamW_CASA4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CASA4.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CASA34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CASA34.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CASA234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CASA234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CASA12344 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-9.3.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CASA12344.py --amp"











































