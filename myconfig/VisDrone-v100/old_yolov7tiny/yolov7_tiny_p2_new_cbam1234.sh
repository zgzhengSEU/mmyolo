# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/S20zhengzg/soft/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/S20zhengzg/soft/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/S20zhengzg/soft/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/S20zhengzg/soft/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate mmyolo
cd ~/mmyolo

set -x

CONFIG_FILE="myconfig/VisDrone-v100/yolov7_tiny_p2_new_cbam1234.py"

name="yolov7_tiny-"
Wandb_NameALL="visualizer.vis_backends.1.init_kwargs.name=${name}"

tags="['v100','yolov7_tiny','300e','amp','p2','v5-kmeans','cbam1234','autoSGD']"
Wandb_TagsALL="visualizer.vis_backends.1.init_kwargs.tags=\"${tags}\""

batch_size="train_dataloader.batch_size="

# 单GPU训练
# python tools/train.py $CONFIG_FILE --cfg-options ${Wandb_NameALL}4 ${Wandb_TagsALL} ${batch_size}4

CUDA_VISIBLE_DEVICES=1 python tools/train.py $CONFIG_FILE --amp --cfg-options ${Wandb_NameALL}32-V100-v5kmeans-p2-cbam1234-autoSGD ${Wandb_TagsALL}

