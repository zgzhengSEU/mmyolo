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

CONFIG_FILE="/home/S20zhengzg/mmyolo/myconfig/VisDrone-v100/yolov5s/yolo.py"


python tools/train.py $CONFIG_FILE 2>&1 >> yoloyolorun.log &