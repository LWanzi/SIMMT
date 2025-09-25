import argparse
import os
import time
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import cv2
from utils import clustering
from MMD import TrainNanosingleAddcl
import torch
import subprocess
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from preprocess import preprocess_CosMx_data
from sklearn.model_selection import ParameterGrid
# os.environ['R_HOME'] = '/usr/lib/R'### Our R installation path
# os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.3'
# os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Versions/4.2/Resources'
# os.environ['R_HOME'] = '/Library/Frameworks/R.framework/Resources'
# os.environ['R_HOME'] ='/Users/wanzi/anaconda3/envs/Sigra/lib/R'
os.environ['R_HOME'] ='/mnt/sda/sww/yes/envs/py38/lib/R'
os.environ['PYTHONHASHSEED'] = '1234'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
# os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.2/bin:$PATH'
start_total_time = time.time()
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 检查并终止之前的Python后台进程
# def terminate_previous_python_processes():
#     os.system("pkill -f 'python train.py'")
# ids = [
#     'fov1']
# img_names = [
#     'F001']
ids = [
     'fov1', 'fov2', 'fov3', 'fov4', 'fov5',
     'fov6', 'fov7', 'fov8', 'fov9', 'fov10',
     'fov11', 'fov12', 'fov13', 'fov14', 'fov15',
     'fov16', 'fov17', 'fov18', 'fov19', 'fov20'
]
img_names = [
     'F001', 'F002', 'F003', 'F004', 'F005',
     'F006', 'F007', 'F008', 'F009', 'F010',
     'F011', 'F012', 'F013', 'F014', 'F015',
     'F016', 'F017', 'F018', 'F019', 'F020',
]
#ids = [
#    'fov6', 'fov8', 'fov10', 'fov12'
#]
#img_names = [
#    'F006', 'F008', 'F010', 'F012'
#]
import time
start_total_time = time.time()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='nanostring', help='should be nanostring, 10x or merscope')#nanostring 10x
    parser.add_argument('--lr', type=float, default=1e-3)#1e-3  default
    parser.add_argument('--root', type=str, default='../dataset/nanostring')#nanostring  DLPFC
    parser.add_argument('--epochs', type=int, default=900)#900
    #parser.add_argument('--id', type=str, default='fov2')#nanostring --fov1   DLPFC --151507
    #parser.add_argument('--img_name', type=str, default='F002')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_path', type=str, default='../checkpoint/nanostring_train')#nanostring_train  10x_train
    parser.add_argument('--ncluster', type=int, default=8)#7
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--use_gray', type=float, default=0)
    parser.add_argument('--test_only', type=int, default=0)#0为训练，1为测试
    parser.add_argument('--pretrain', type=str, default='all/final_900_0.pth')#nanostring --all/final_900_0.pth  DLPFC --final.pth
    parser.add_argument('--cluster_method', type=str, default='leiden', help='leiden or mclust')

    opt = parser.parse_args()


    for id, name in zip(ids, img_names):
        start_time = time.time()  # 每个模型的训练开始时间
        sp = os.path.join(opt.save_path,id)
        if not os.path.exists(sp):
            os.makedirs(sp)

        adata= preprocess_CosMx_data(opt, id, name)#, patchs, centers
        model = TrainNanosingleAddcl(adata,hidden_dims=[16], n_epochs=opt.epochs, lr=opt.lr,
                                    random_seed=opt.seed, save_path=sp, ncluster=opt.ncluster, repeat=1,temperature=0.55)  #  patchs, centers,
        # train model
        adata = model.train()
        # 记录当前id的结束时间
        end_time = time.time()
        model_running_time = end_time - start_time  # 当前id的运行时间
        print(f"Model {id} running time: {model_running_time:.2f} seconds")
        end_total_time = time.time()
        total_running_time = end_total_time - start_total_time
        print("Total running time:", total_running_time)
        # adata.write("/cosmx.h5ad")
        # clustering
        result = clustering(adata, opt.ncluster, id)
        print('ari is %.2f' % (result))
        print("##########################")

        # 生成输出文件名，包含当前时间戳
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"{id}_{timestamp}.log"
        result_file="result.log"
        # 将结果保存到文件
        with open(result_file, "a") as file:
            file.write(f"{id}:{result:.2f}\n")
        # 使用 subprocess 异步执行 nohup 命令
        command = f"nohup python train.py {id} >> {output_file} 2>&1 &"
        subprocess.Popen(command, shell=True)
        # # 使用nohup命令在后台运行，将输出追加到总日志文件
        # command = f"nohup python train.py {id} >> {output_file} 2>&1 &"
        # os.system(command)
