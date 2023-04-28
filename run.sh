bsub -J yolov7_tiny_sgd64 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_sgd64.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_TA1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_TA1234.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_TA234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_TA234.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_TA34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_TA34.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_SIOU.py --amp"


bsub -J yolov7_tiny_tinyp2_sgd64_ASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_ASFF.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_ASFFCE -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_ASFFCE.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_ASFFsim -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_ASFFsim.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_ASFFsimCE -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_ASFFsimCE.py --amp"

bsub -J yolov7_tiny_sgd64_load -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_sgd64_load.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_v6loss -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_v6loss.py --amp"


# ======================================================
bsub -J yolov7_tiny_tinyp2_AdamW_TA4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_TA4.py --amp"

# ===============================================

bsub -J yolov7_tiny_tinyp2_AdamW_CA4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA34.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA1234.py --amp"

# =========================

bsub -J yolov7_tiny_tinyp2_AdamW_SA4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SA4.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SA34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SA34.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SA234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SA234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SA1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SA1234.py --amp"

# ==========================



bsub -J yolov7_tiny_tinyp2_AdamW_CASA4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CASA4.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CASA34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CASA34.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CASA234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CASA234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CASA1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CASA1234.py --amp"


# ======================

bsub -J yolov7_tiny_tinyp2_AdamW_VFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_VFL-SIOU.py --amp"


# ===================

bsub -J yolov7_tiny_tinyp2_AdamW_TA1234-SA1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_TA1234-SA1234.py --amp"


# ======================

bsub -J yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234g8_ASFFsimCE_QFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234g8_ASFFsimCE_QFL-SIOU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234g8_ASFFsimCE_VFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_SiLU_TA1234-SA1234g8_ASFFsimCE_VFL-SIOU.py --amp"


# ===================== CSA

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g4.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_SIOU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_SiLU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_SiLU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SIOU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_SIOU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_SIOU_v6loss -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_SIOU_v6loss.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_SIOU_v6loss_bs4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_SIOU_v6loss_bs4.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_SIOU_v6loss_bs2 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_SIOU_v6loss_bs2.py --amp"


# ======= 

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_VFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_VFL-SIOU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_VFL-CIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_VFL-CIOU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_VFL-SIOU_500e -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_VFL-SIOU_500e.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_VFLnew-CIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_ASFFsimCE_SiLU_VFLnew-CIOU.py --amp"



# ======================= new ==================================

bsub -J yolov7_tiny_AdamW -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_AdamW.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TinyCEASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_TinyCEASFF.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_TinyCEASFF_VFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_TinyCEASFF_VFL-SIOU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_TinyCEASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8_TinyCEASFF.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g8.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CA4r16-SA1234g4.py --amp"


# ====================== new CEPAFPN ===============================
bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_TinyCEASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_TinyCEASFF.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_CA4r16-SA1234g8_TinyCEASFF_VFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_CA4r16-SA1234g8_TinyCEASFF_VFL-SIOU.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_CA4r16-SA1234g8_TinyCEASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_CA4r16-SA1234g8_TinyCEASFF.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_CA4r16-SA1234g8 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_CA4r16-SA1234g8.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_CA4r16-SA1234g4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_CA4r16-SA1234g4.py --amp"

# SCA zyz
bsub -J yolov7_tiny_tinyp2_AdamW_SCAg8-4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200858/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg8-4.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SCAg8-34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200858/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg8-34.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SCAg8-234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200858/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg8-234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SCAg8-1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200858/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg8-1234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SCAg16-4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200858/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg8-4.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SCAg16-34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200858/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg8-34.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SCAg16-234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200858/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg8-234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_SCAg16-1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200858/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg8-1234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200858/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200858/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF.py --amp"

# SCA zzg
bsub -J yolov7_tiny_tinyp2_AdamW_SCAg8-4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg8-4.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_SCAg8-34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg8-34.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_SCAg8-234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg8-234.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_SCAg8-1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg8-1234.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_SCAg16-4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg16-4.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_SCAg16-34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg16-34.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_SCAg16-234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg16-234.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_SCAg16-1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg16-1234.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF.py --amp"


# ====

bsub -J yolov7_tiny_tinyp2_AdamW_SCAg4-4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg4-4.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_SCAg4-34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg4-34.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_SCAg4-234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg4-234.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_SCAg4-1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_SCAg4-1234.py --amp"


bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF_VFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg8-1234_TinyCEASFF_VFL-SIOU.py --amp"


bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg4-1234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg4-1234.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg4-1234_TinyCEASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg4-1234_TinyCEASFF.py --amp"
bsub -J yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg4-1234_TinyCEASFF_VFL-SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_CEPAFPN_SCAg4-1234_TinyCEASFF_VFL-SIOU.py --amp"


# ==== tinysppf zzg
bsub -J yolov7_tiny_tinyp2_AdamW_TinySPPF-g4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_TinySPPF-g4.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TinySPPF-g8 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_TinySPPF-g8.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TinySPPF-g16 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_TinySPPF-g16.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TinyCEASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_TinyCEASFF.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TinyCEASFF-g4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_TinyCEASFF-g4.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TinyCEASFF-g8 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_TinyCEASFF-g8.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TinyCEASFF-g16 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_TinyCEASFF-g16.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TinyCEASFF-g8x2 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_TinyCEASFF-g8x2.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TinyCEASFF-g16x2 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_TinyCEASFF-g16x2.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_TinyCEASFF-g32x2 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/yolov7_tiny_new/yolov7_tiny_tinyp2_AdamW_TinyCEASFF-g32x2.py --amp"












# =========================================== only test ==================
bsub -J install-carafe -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yufei/220200817/.conda/envs/mmyolo/bin/python setup.py develop"

bsub -J install-carafe -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python setup.py develop"

bsub -J install-carafe -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && make install && make test"

# ============================ Final ===================================


bsub -J yolov7_tiny_originsgd_Mish -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/Final/yolov7_tiny_originsgd_Mish.py --amp"

bsub -J yolov7_tiny_originsgd_QFL -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/Final/yolov7_tiny_originsgd_QFL.py --amp"

bsub -J yolov7_tiny_originsgd_SCAg4 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/Final/yolov7_tiny_originsgd_SCAg4.py --amp"

bsub -J yolov7_tiny_originsgd_TinyASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/Final/yolov7_tiny_originsgd_TinyASFF.py --amp"

bsub -J yolov7_tiny_originsgd -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/Final/yolov7_tiny_originsgd.py --amp"

bsub -J yolov7_tiny_tinyp2_originsgd -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/Final/yolov7_tiny_tinyp2_originsgd.py --amp"

bsub -J yolov7_tiny_tinyp2_originsgd_Mish_SCAg4_TinyASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/Final/yolov7_tiny_tinyp2_originsgd_Mish_SCAg4_TinyASFF.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_Mish_SCAg4_TinyASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/Final/yolov7_tiny_tinyp2_AdamW_Mish_SCAg4_TinyASFF.py --amp"

bsub -J yolov7_tiny_tinyp2_AdamW_Mish_SCAg4_TinyASFF_QFL -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/Final/yolov7_tiny_tinyp2_AdamW_Mish_SCAg4_TinyASFF_QFL.py --amp"

bsub -J yolov7_tiny_tinyp2_originsgd_Mish_SCAg4_TinyASFF_QFL -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3 && source activate mmyolo && module load cuda-11.6 && module load gcc-7.5.0 && /seu_share/home/yanglvxi/220200815/.conda/envs/mmyolo/bin/python tools/train.py myconfig/VisDrone-seu/Final/yolov7_tiny_tinyp2_originsgd_Mish_SCAg4_TinyASFF_QFL.py --amp"










