
bsub -J yolov7_tiny_AdamW -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_AdamW.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TA1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_TA1234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TA234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_TA234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TA34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_TA34.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SIOU.py --amp"


bsub -J yolov7_tiny_tinyp2_AdamW_ASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_ASFF.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_ASFFCE -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_ASFFCE.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_ASFFsim -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_ASFFsim.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_ASFFsimCE -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_ASFFsimCE.py --amp"


# ======================


bsub -J yolov7_tiny_tinyp2_AdamW_CASA234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CASA234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CASA1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CASA1234.py --amp"

# ================================

bsub -J yolov7_tiny_tinyp2_AdamW_noload -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_noload.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW64x2_noload -q gpu_v100 -gpu "num=2:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && bash ./tools/dist_train.sh myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW64x2_noload.py 2 --amp"


# ================================

bsub -J yolov7_tiny_tinyp2_sgd64x2 -q gpu_v100 -gpu "num=2:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && bash ./tools/dist_train.sh myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64x2.py 2 --amp"

bsub -J yolov7_tiny_tinyp2_sgd64x2_noload -q gpu_v100 -gpu "num=2:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && bash ./tools/dist_train.sh myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64x2_noload.py 2 --amp"

# ================

bsub -J yolov7_tiny_tinyp2_sgd64x2_noload_CASA1234_ASFFsimCE -q gpu_v100 -gpu "num=2:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && bash ./tools/dist_train.sh myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64x2_noload_CASA1234_ASFFsimCE 2 --amp"
# =================

bsub -J yolov7_tiny_tinyp2_AdamW-TA1234-SA1234_ASFFsimCE_VFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW-TA1234-SA1234_ASFFsimCE_VFL-SIOU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW-TA1234-SA1234_ASFFsimCE_VFL-SIOU_noload -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW-TA1234-SA1234_ASFFsimCE_VFL-SIOU_noload.py --amp"

# =================== SiLU

bsub -J yolov7_tiny_tinyp2_AdamW_SiLU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SiLU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234_ASFFsimCE_VFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234_ASFFsimCE_VFL-SIOU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234_ASFFsimCE_QFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234_ASFFsimCE_QFL-SIOU.py --amp"

bsub -J yolov7_tiny_AdamW_SiLU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_AdamW_SiLU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SiLU_SA1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SiLU_SA1234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SiLU_TA1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SiLU_TA1234.py --amp"


# ================== QFL
bsub -J yolov7_tiny_tinyp2_AdamW-QFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW-QFL-SIOU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW-TA1234-SA1234_ASFFsimCE_QFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW-TA1234-SA1234_ASFFsimCE_QFL-SIOU.py --amp"


# ====================== gpus
# CONFIG=$1
# NNODES=${NNODES:-1}
# NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29500}
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python -m torch.distributed.launch \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --nproc_per_node=2 \
#     --master_port=$PORT \
#     $(dirname "$0")/train.py \
#     $CONFIG \
#     --launcher pytorch ${@:3}

# ====================== SAg8

bsub -J yolov7_tiny_tinyp2_AdamW_SA1234g8 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SA1234g8.py --amp"

# ================= v6loss
bsub -J yolov7_tiny_tinyp2_AdamW_v6loss -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_v6loss.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TA1234-SA1234_v6loss -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_TA1234-SA1234_v6loss.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TA1234-SA1234_ASFFsimCE_v6loss -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_TA1234-SA1234_ASFFsimCE_v6loss.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234_ASFFsimCE_v6loss -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234_ASFFsimCE_v6loss.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234g8_ASFFsimCE_v6loss -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234g8_ASFFsimCE_v6loss.py --amp"

# ============

bsub -J yolov7_tiny_tinyp2_AdamW_TA1234-SA1234_ASFFsimCE -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_TA1234-SA1234_ASFFsimCE.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_VFL-SIOU_500e -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_VFL-SIOU_500e.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_VFL-CIOU_500e -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_VFL-CIOU_500e.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_VFL-GIOU_500e -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_VFL-GIOU_500e.py --amp"

