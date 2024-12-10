import math
import pandas as pd
import numpy as np
import sklearn.neighbors
import scipy.sparse as sp
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
import torch
import scanpy as sc
from torch_geometric.data import Data
import torch.backends.cudnn as cudnn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Cal_Spatial_Net(adata, rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True, use_global=False):  # 构建空间网络
    if verbose:
        print('------Calculating spatial graph...')
    if use_global:  # 获取坐标数据
        coor = pd.DataFrame(adata.obsm['spatial_global'])
    else:
        coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    # 基于半径的最近邻居搜索
    nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
    distances, indices = nbrs.radius_neighbors(coor, return_distance=True)  # 获取每个细胞的邻居信息，包括邻居的索引和距离
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']  # 将邻居信息存储在 KNN_df中，包括细胞1的索引、细胞2的索引和它们之间的距离

    Spatial_Net = KNN_df.copy()
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]  # 通过复制KNN_df创建 Spatial_Net，然后筛选掉距离为0的边来确保不包含自环
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)  # 将索引转换为对应的细胞标识
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))

    adata.uns['Spatial_Net'] = Spatial_Net  # 将构建的空间网络存储在 adata.uns['Spatial_Net'] 中


def Stats_Spatial_Net(adata):  # 计算和可视化细胞之间邻居数量的分布，提供了关于数据集空间网络特征的统计信息
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / adata.shape[0]
    fig, ax = plt.subplots(figsize=[3, 2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.bar(plot_df.index, plot_df)
    plt.close('all')


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='pred', random_seed=0):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    # b=rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames,
                  verbose=False)  # for using pre_trained checkpoint
    # res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata[0].obsm[used_obsm]), num_cluster, modelNames, verbose=False)#for training from scratch
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    # adata[0].obs['mclust'] = mclust_res
    # adata[0].obs['mclust'] = adata[0].obs['mclust'].astype('int')
    # adata[0].obs['mclust'] = adata[0].obs['mclust'].astype('category')
    return adata


def _hungarian_match(flat_preds, flat_target, preds_k, target_k):
    num_samples = flat_target.shape[0]
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))
    for c1 in range(num_k):
        for c2 in range(num_k):
            votes = int(((flat_preds == c1) * (flat_target == c2)).sum())
            num_correct[c1, c2] = votes

    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))
    return res
def prefilter_genes(adata,min_counts=None,max_counts=None,min_cells=10,max_cells=None):
    if min_cells is None and min_counts is None and max_cells is None and max_counts is None:
        raise ValueError('Provide one of min_counts, min_genes, max_counts or max_genes.')
    id_tmp=np.asarray([True]*adata.shape[1],dtype=bool)
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_cells=min_cells)[0]) if min_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_cells=max_cells)[0]) if max_cells is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,min_counts=min_counts)[0]) if min_counts is not None  else id_tmp
    id_tmp=np.logical_and(id_tmp,sc.pp.filter_genes(adata.X,max_counts=max_counts)[0]) if max_counts is not None  else id_tmp
    adata._inplace_subset_var(id_tmp)
    return id_tmp, adata


def prefilter_specialgenes(adata,Gene1Pattern="ERCC",Gene2Pattern="MT-"):
    id_tmp1=np.asarray([not str(name).startswith(Gene1Pattern) for name in adata.var_names],dtype=bool)
    id_tmp2=np.asarray([not str(name).startswith(Gene2Pattern) for name in adata.var_names],dtype=bool)
    id_tmp=np.logical_and(id_tmp1,id_tmp2)
    adata._inplace_subset_var(id_tmp)
    return id_tmp, adata


def clustering(adata, ncluster,id,key='emb', method='mclust', start=0.1, end=3.0, increment=0.01,refinement=False):
    if method == 'mclust':
        pca = PCA(n_components=15, random_state=42)
        embedding = pca.fit_transform(adata.obsm['pred'].copy())
        adata.obsm['pred_pca'] = embedding
        # adata = mclust_R(adata, used_obsm='pred', num_cluster=opt.ncluster)
        adata = mclust_R(adata, used_obsm='pred_pca', num_cluster=ncluster)
        # obs_df = adata[0].obs.dropna()
        # sc.pl.spatial(adata, color="Ground Truth")##true label
        obs_df = adata.obs.dropna()
        # 生成保存路径
        save_path = f"/{id}.png"#figures/show
        # sc.pl.spatial(adata, color="mclust", spot_size=10) ##predicted label        ,show=True
        # sc.pl.spatial(adata, color="mclust", spot_size=10, save=save_path)
        # sc.pl.spatial(adata, color="merge_cell_type", spot_size=70, save=save_path)#10   merge_cell_type  mclust
        #！假设1对应lymphocyte，但其实并不一定
        #num_to_cell_type = {1: 'lymphocyte', 2: 'neutrophil', 3: 'mast', 4: 'endothelial',5: 'fibroblast',6: 'epithelial',7: 'Mcell',8: 'tumors'}  # 根据您的实际情况定义映射关系
        # 创建一个新列，将 mclust 标签映射为细胞类型
        #adata.obs['pred_cell_type'] = adata.obs['mclust'].map(num_to_cell_type)
        adata.obs['pred_cell_type'] = adata.obs['mclust']
        # 定义细胞类型名称，主要是定义顺序，该顺序将与颜色板顺序一一对应
        #custom_order = ["lymphocyte", "Mcell", "tumors", "epithelial", "mast", "endothelial", "fibroblast",
                        #"neutrophil"]

        # 2:自定义颜色板，给定颜色代码，要求这些颜色代码为matplotlib支持
        custom_colors = ["#E57272FF", "#FFCA27FF", "#A6CEE3", "#D3E057FF", "#5B6BBFFF", "#26C5D9FF", "#26A599FF",
                         "#B967C7FF"]
        # 将细胞类型与颜色一一对应，生成字典
        # color_fine = dict(zip(list(custom_order), custom_colors))
        # sc.pl.spatial(adata, color="pred_cell_type", spot_size=60, save=save_path)  ##predicted label
        sc.pl.spatial(adata, color="pred_cell_type", palette=custom_colors, spot_size=50,save=save_path)#palette指定细胞类型对应特定的颜色
        # cell_type_to_num = {'lymphocyte': 1, 'neutrophil': 2, 'mast': 3,'endothelial': 4,'fibroblast': 5,'epithelial': 6,'Mcell': 7,'tumors': 8}
        # gt=adata.obs['merge_cell_type'].map(cell_type_to_num)#将真实细胞类型名称与数字对应
        # pred = adata.obs['mclust'].to_numpy().astype(np.int32)
        adata.obsm['imgs'] = adata.obsm['imgs'].to_numpy()
        adata.obsm['coor'] = adata.obsm['coor'].to_numpy()


        # sc.pl.spatial(adata, color="gtcmap", spot_size=60, save=save_path)

        # ARI = adjusted_rand_score(obs_df['mclust'], obs_df['Ground Truth'])
        ARI = adjusted_rand_score(obs_df['mclust'], obs_df['merge_cell_type'])
        # print('ari is %.2f' % (ARI))
        adata.write('../results/lung/we_{0}.h5ad'.format(id))
    return ARI



