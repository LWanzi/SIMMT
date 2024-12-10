import torch
import numpy as np
from torch_geometric.data import Data
import scanpy as sc
import pandas as pd
import random
import os
import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import scipy
import tqdm
from utils import prefilter_genes,prefilter_specialgenes
from utils import Cal_Spatial_Net, Stats_Spatial_Net

def processing_10x(root,sub_id):
    # sub_id = ['151507', '151508', '151509', '151510',
    #         '151669', '151670', '151671', '151672',
    #         '151673', '151674', '151675', '151676']

    # sub_id = ['151673']#151507
    for id in tqdm.tqdm(sub_id):
        dirs = os.path.join(root, id)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        adata = sc.read_visium(dirs, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()
        # 2. filter genes first
        idx1, adata = prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
        idx2, adata = prefilter_specialgenes(adata)

        # 3. save the raw data as csv
        adata.write(os.path.join(dirs, 'sampledata.h5ad'))
        sample_csv = pd.DataFrame(adata.X.toarray(), index=adata.obs.index, columns=adata.var.index)
        sample_csv.to_csv(os.path.join(dirs, 'sample_data.csv'))

def preprocess_adj(edge_index, num_nodes):
    #convert edge_index to adj
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    # edge_df=pd.DataFrame({'source':edge_index[0].cpu().numpy(),'target':edge_index[1].cpu().numpy()})
    # edge_df.to_csv('edge_df.csv',index=True)
    #Laplacian normalize adj
    noramalized_adj=normalize_adj(adj)
    return noramalized_adj
def normalize_adj(adj):#Laplacian normalization
    adj=adj+torch.eye(adj.shape[0])
    row_sum=adj.sum(dim=1)#degree
    D_inv_sqrt=torch.pow(row_sum,-0.5)#degree -1/2
    s=np.isinf(D_inv_sqrt)
    s=s.bool()
    # print(torch.is_bool_tensor(s))
    # D_inv_sqrt[np.isinf(D_inv_sqrt)]=0.0#deal with degree with 0
    D_inv_sqrt[s] = 0.0  # deal with degree with 0
    D_inv_sqrt=torch.diag(D_inv_sqrt)#diag matrix
    adj=torch.matmul(torch.matmul(D_inv_sqrt,adj),D_inv_sqrt)

    return adj

def Transfer_img_Data_DLPFC(adata):  # 将从adata对象中提取的空间网络数据转换为PyTorch中的Data对象
    G_df = adata.uns['Spatial_Net'].copy()  # 读取空间网络数据，保留两个细胞名称以及之间的距离
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))  # 构建了一个映射字典cells_id_tran即cell 1:0,cell2 :1
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)  # 将细胞名字cells映射为从0开始的整数
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    e0 = G_df['Cell1'].to_numpy()
    e1 = G_df['Cell2'].to_numpy()
    edgeList = np.array((e0, e1))  # 将映射后的细胞构建了边的列表edgeList，cell1索引，cell2索引
    data = Data(edge_index=torch.LongTensor(np.array(
        [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    img = Data(edge_index=torch.LongTensor(np.array(
        [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['imgs'].to_numpy()))
    # return data, img,adata.obsm['imgs'],adata.obsm['coor']   for cosmx
    return data, img  # for dlpfc
def gen_adatas(root, id, img_name):
    adata = sc.read(os.path.join(root, id, 'sampledata.h5ad'))#一个fov对应的基因表达数据
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    ncluster = len(set(adata.obs['merge_cell_type']))
    print(os.path.join(root, id, 'CellComposite_%s.jpg' % (img_name)))
    img = cv2.imread(os.path.join(root, id, 'CellComposite_%s.jpg' % (img_name)))#读取图像数据并返回一个多维数组：(3648,5472,3)
    height, width, c = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#将一个BGR格式的图像（OpenCV中默认的颜色通道顺序是Blue、Green、Red）转换为RGB格式，图像的内容不发生变化，只是颜色通道的排列顺序改变了
    overlay = cv2.imread(os.path.join(root, id, 'CompartmentLabels_%s.tif'%(img_name)))#读取图像标签
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)#将一个BGR格式的图像转换为灰度图像，2维
    print(overlay.shape)
    # if opt.use_gray:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.ToTensor()
    img = transform(img)#数组转换成向量
    overlay = transform(overlay)
    patchs = []
    w, h = 60, 60
    centers=[]
    for coor in adata.obsm['spatial']:
        x, y = coor#coor细胞的空间位置坐标
        img_p = img[:, int(y-h):int(y+h), int(x-w): int(x+w)]#在img的三个维度上进行切片，在第一个维度上取所有的元素，即不进行切片，在第二个维度上取从 y-h 到 y+h 范围内的像素，在第三个维度上取从 x-w 到 x+w 范围内的像
        #从img图像中截取一个以(x, y)为中心、宽度为2w、高度为2h的矩形区域即patch，捕获图像局部信息
        patchs.append(img_p.flatten()) # (cell number,3 * 2h * 2w )  orig
        # patchs.append(img_p)
        centers.append(coor)
    patchs = np.stack(patchs)# 数组
    centers = np.stack(centers)

    df = pd.DataFrame(patchs, index=adata.obs.index)#将细胞与对应的图像块patch对应,存储到imgs
    adata.obsm['imgs'] = df

    d1 = pd.DataFrame(centers, index=adata.obs.index)#将细胞与对应的图像块patch对应,存储到imgs
    adata.obsm['coor'] = d1

    Cal_Spatial_Net(adata, rad_cutoff=80)#构建空间网络
    Stats_Spatial_Net(adata)#统计空间网络特征信息
    return adata
def preprocess_CosMx_data(opt,id,name):
    ##preprocess data get sampledata.h5ad   optional
    # processing_nano(opt.root,sub_id=opt.id)
    # adata = gen_adatas(opt.root, opt.id, opt.img_name)  # 构建空间网络，并将信息存储到adata(包括patch)
    adata = sc.read(os.path.join(opt.root, id, 'sampledata.h5ad'))  # 一个fov对应的基因表达数据
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)##new

    ncluster = len(set(adata.obs['merge_cell_type']))
    print(os.path.join(opt.root, id, 'CellComposite_%s.jpg' % (name)))
    img = cv2.imread(os.path.join(opt.root, id, 'CellComposite_%s.jpg' % (name)))  # 读取图像数据并返回一个多维数组：(3648,5472,3)
    height, width, c = img.shape
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)  # 将一个BGR格式的图像（OpenCV中默认的颜色通道顺序是Blue、Green、Red）转换为RGB格式，图像的内容不发生变化，只是颜色通道的排列顺序改变了
    overlay = cv2.imread(os.path.join(opt.root, id, 'CompartmentLabels_%s.tif' % (name)))  # 读取图像标签
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)  # 将一个BGR格式的图像转换为灰度图像，2维
    # print(overlay.shape)
    # if opt.use_gray:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    transform = transforms.ToTensor()
    img = transform(img)  # 数组转换成向量
    overlay = transform(overlay)
    patchs = []
    w, h = 60, 60
    centers = []
    for coor in adata.obsm['spatial']:
        x, y = coor  # coor细胞的空间位置坐标
        img_p = img[:, int(y - h):int(y + h), int(x - w): int(
            x + w)]  # 在img的三个维度上进行切片，在第一个维度上取所有的元素，即不进行切片，在第二个维度上取从 y-h 到 y+h 范围内的像素，在第三个维度上取从 x-w 到 x+w 范围内的像
        # 从img图像中截取一个以(x, y)为中心、宽度为2w、高度为2h的矩形区域即patch，捕获图像局部信息
        patchs.append(img_p.flatten())  # (cell number,3 * 2h * 2w )  orig
        # patchs.append(img_p)
        centers.append(coor)
    patchs = np.stack(patchs)  # 数组
    centers = np.stack(centers)

    df = pd.DataFrame(patchs, index=adata.obs.index)  # 将细胞与对应的图像块patch对应,存储到imgs
    adata.obsm['imgs'] = df

    d1 = pd.DataFrame(centers, index=adata.obs.index)  # 将细胞与对应的图像块patch对应,存储到imgs
    adata.obsm['coor'] = d1

    Cal_Spatial_Net(adata, rad_cutoff=80)  # 构建空间网络
    Stats_Spatial_Net(adata)  # 统计空间网络特征信息
    return adata,patchs,centers
def Transfer_img_Data(adata):#将从adata对象中提取的空间网络数据转换为PyTorch中的Data对象
    G_df = adata.uns['Spatial_Net'].copy()#读取空间网络数据，保留两个细胞名称以及之间的距离
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))#构建了一个映射字典cells_id_tran即cell 1:0,cell2 :1
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)#将细胞名字cells映射为从0开始的整数
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    # print(G_df)
    # exit(0)
    e0 = G_df['Cell1'].to_numpy()
    e1 = G_df['Cell2'].to_numpy()
    edgeList = np.array((e0, e1))#将映射后的细胞构建了边的列表edgeList，cell1索引，cell2索引
    data = Data(edge_index=torch.LongTensor(np.array(
        [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.X.todense()))  # .todense()
    img = Data(edge_index=torch.LongTensor(np.array(
        [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['imgs'].to_numpy()))
    return data, img
def adj_to_edge_index(adj_matrix):
    edge_index = []
    num_nodes = len(adj_matrix)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                # 将节点 i 到节点 j 的边添加到边索引中
                edge_index.append([i, j])

    return edge_index