bsub -J yolov7_tiny_sgd64 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3;module load cuda-11.6;module load gcc-9.3.0;source activate mmyolo;cd ~/mmyolo;python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_sgd64.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3;module load cuda-11.6;module load gcc-9.3.0;source activate mmyolo;cd ~/mmyolo;python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_v6loss -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3;module load cuda-11.6;module load gcc-9.3.0;source activate mmyolo;cd ~/mmyolo;python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_v6loss.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_TA234 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3;module load cuda-11.6;module load gcc-9.3.0;source activate mmyolo;cd ~/mmyolo;python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_TA234.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_TA34 -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3;module load cuda-11.6;module load gcc-9.3.0;source activate mmyolo;cd ~/mmyolo;python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_TA34.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_SIOU -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3;module load cuda-11.6;module load gcc-9.3.0;source activate mmyolo;cd ~/mmyolo;python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_SIOU.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_ASFF -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3;module load cuda-11.6;module load gcc-9.3.0;source activate mmyolo;cd ~/mmyolo;python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_ASFF.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_ASFFCE -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3;module load cuda-11.6;module load gcc-9.3.0;source activate mmyolo;cd ~/mmyolo;python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_ASFFCE.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_ASFFsim -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3;module load cuda-11.6;module load gcc-9.3.0;source activate mmyolo;cd ~/mmyolo;python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_ASFFsim.py --amp"

bsub -J yolov7_tiny_tinyp2_sgd64_ASFFsimCE -q gpu_v100 -gpu "num=1:mode=exclusive_process:aff=yes" "module load anaconda3;module load cuda-11.6;module load gcc-9.3.0;source activate mmyolo;cd ~/mmyolo;python tools/train.py myconfig/VisDrone-seu/yolov7_tiny/yolov7_tiny_tinyp2_sgd64_ASFFsimCE.py --amp"


