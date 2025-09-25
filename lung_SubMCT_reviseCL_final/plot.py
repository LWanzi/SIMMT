import scanpy as sc
import os
import seaborn as sns
import pandas as pd
from utils import mclust_R
import matplotlib.pyplot as plt
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
os.environ['R_HOME'] = 'C:\Program Files\R\R-4.3.3'
root='../dataset/nanostring'
for id, name in zip(ids, img_names):
    adata = sc.read(os.path.join(root, id, 'sampledata.h5ad'))
    # adata = mclust_R(adata, used_obsm='pred_pca', num_cluster=8)
    # 生成保存路径
    save_path = f"/{id}.png"  # figures/show
    # colors = {'lymphocyte': '#E57272FF',
    #           'Mcell': '#FFCA27FF',
    #           'tumors': '#A6CEE3',
    #           'epithelial': '#D3E057FF',
    #           'mast': '#5B6BBFFF',
    #           'endothelial': '#26C5D9FF',
    #           'fibroblast': '#26A599FF',
    #           'neutrophil': '#B967C7FF'
    #           }
    # Set colors used

    # ax.axes.invert_yaxis()
    # plt.savefig("./sample_results/pred.png", dpi=600)
    # plt.close()

    # 定义细胞类型名称，主要是定义顺序，该顺序将与颜色板顺序一一对应
    custom_order = ["lymphocyte", "Mcell", "tumors", "epithelial", "mast", "endothelial", "fibroblast",
                    "neutrophil"]
    # 2:自定义颜色板，给定颜色代码，要求这些颜色代码为matplotlib支持
    custom_colors = ["#E57272FF", "#FFCA27FF", "#A6CEE3", "#D3E057FF", "#5B6BBFFF", "#26C5D9FF", "#26A599FF",
                     "#B967C7FF"]
    # 将细胞类型与颜色一一对应，生成字典
    color_fine = dict(zip(list(custom_order), custom_colors))
    # sc.pl.spatial(adata, color="pred_cell_type", spot_size=60, save=save_path)  ##predicted label
    sc.pl.spatial(adata, color="merge_cell_type", palette=color_fine, spot_size=50,save=save_path)  # palette指定细胞类型对应特定的颜色



