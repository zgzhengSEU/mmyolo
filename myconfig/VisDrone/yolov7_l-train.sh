# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/gp.sc.cc.tohoku.ac.jp/duanct/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/gp.sc.cc.tohoku.ac.jp/duanct/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/gp.sc.cc.tohoku.ac.jp/duanct/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/gp.sc.cc.tohoku.ac.jp/duanct/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate mmyolo
cd ~/zhengzg/mmyolo

set -x

CONFIG_FILE="/home/gp.sc.cc.tohoku.ac.jp/duanct/zhengzg/mmyolo/myconfig/VisDrone/yolov7_l_syncbn_fast_8x16b-300e_coco.py"

name="yolov7_l-"
Wandb_NameALL="visualizer.vis_backends.1.init_kwargs.name=${name}"

tags="['h100','yolov7_l','300e']"
Wandb_TagsALL="visualizer.vis_backends.1.init_kwargs.tags=\"${tags}\""

batch_size="train_dataloader.batch_size="

# 单GPU训练
# python tools/train.py $CONFIG_FILE --cfg-options ${Wandb_NameALL}4 ${Wandb_TagsALL} ${batch_size}4

# python tools/train.py $CONFIG_FILE --cfg-options ${Wandb_NameALL}8 ${Wandb_TagsALL} ${batch_size}8

# python tools/train.py $CONFIG_FILE --cfg-options ${Wandb_NameALL}16 ${Wandb_TagsALL} ${batch_size}16

# python tools/train.py $CONFIG_FILE --cfg-options ${Wandb_NameALL}24 ${Wandb_TagsALL} ${batch_size}24

python tools/train.py $CONFIG_FILE --cfg-options ${Wandb_NameALL}32 ${Wandb_TagsALL} ${batch_size}32

# python tools/train.py $CONFIG_FILE --cfg-options ${Wandb_NameALL}40 ${Wandb_TagsALL} ${batch_size}40

# python tools/train.py $CONFIG_FILE --cfg-options ${Wandb_NameALL}48 ${Wandb_TagsALL} ${batch_size}48