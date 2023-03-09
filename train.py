import os
import sys
import time
from multiprocessing import  Process
import socket
import argparse
import random
gpu_id = 'CUDA_VISIBLE_DEVICES='
# cmd = '~/soft/anaconda3/envs/open-mmlab/python train.py'

def grap_single_gpu(interval, gpu_index, cmd):
    hostname = socket.gethostname()
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    need_gpu_memory = 12
    while meminfo.free/(1024**2) < need_gpu_memory*1024:
        GPUBUSY = hostname + " GPU " + str(gpu_index) + " BUSY " + str(meminfo.free/(1024**3))[0:3] + "GB " + time.strftime("%H:%M:%S", time.localtime()) 
        sys.stdout.write('\r' + GPUBUSY)
        sys.stdout.flush()
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        time.sleep(interval)
    
    print('\n\n\nGPU'+str(gpu_index)+' FREE '+str(meminfo.free/(1024**3))+'GB' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) 
    # time.sleep(10)
    print(gpu_id + str(gpu_index) + ' ' + str(cmd))
    os.system("cd")
    os.system(gpu_id + str(gpu_index) + ' ' + str(cmd))    


if __name__ == '__main__':
    process_list = []
    hostname = socket.gethostname()
    if hostname == "mu01":
        gpu_num = 2
    elif hostname == "gpu01":
        gpu_num = 4
    else:
        gpu_num = 6

    parser = argparse.ArgumentParser()
    parser.add_argument('start', type=int, default=0)
    parser.add_argument('end', type=int, default=gpu_num - 1)
    parser.add_argument('--cmd',type=str, nargs='?', const="auto")
    args = parser.parse_args()
    if args.cmd == 'auto':
        cmd = '~/soft/anaconda3/envs/open-mmlab/python train.py'
    else:
        cmd = args.cmd
    print(args.start)
    print(args.end)
    print(cmd) 
    factor = 0.2
    interval = gpu_num * factor
    for i in range(args.start, args.end + 1):  #开启4个子进程执行
        p = Process(target=grap_single_gpu,args=(interval, i, cmd)) #实例化进程对象
        p.start()
        process_list.append(p)
        time.sleep(1 * factor)

    for i in process_list:
        p.join()
    #grap_single_gpu(0)
    print('主程序执行完毕')
