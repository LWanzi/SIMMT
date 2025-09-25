import os
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from torch_geometric.loader import NeighborLoader, NeighborSampler, DataLoader
from sklearn.metrics.cluster import adjusted_rand_score
# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# from torch_geometric.loader import NeighborLoader
from model import TransImg_CosMx
from loss import CL_loss,CL_loss_SCprotein,CL_loss_BLEEP,sce_loss,CL_loss_with_Spatial_graph
from utils import seed_everything, mclust_R
from preprocess import preprocess_adj,Transfer_img_Data
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, SequentialSampler
from torch_geometric.data import Data
import anndata
class TrainNanosingleAddcl(nn.Module):
    def __init__(self,
                 adata,
                 # patchs,
                 # centers,
                 hidden_dims=[32],#[512, 30]
                 n_epochs=900,
                 lr=0.001,
                 key_added='STAGATE',
                 gradient_clipping=5.,
                 weight_decay=0,
                 verbose=True,
                 random_seed=0,
                 save_reconstruction=False,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                 save_path='../checkpoint/trans_gene/',
                 ncluster=8,
                 repeat=1,
                 lambda_1=1,
                 lambda_2=1,
                 lambda_3=1,#1
                 lambda_4=0.1,temperature=0.1):
        super(TrainNanosingleAddcl, self).__init__()  # 继承自 nn.Module
        self.adata = adata
        # self.patchs = patchs
        # self.centers = centers
        self.hidden_dims = hidden_dims
        self.n_epochs = n_epochs
        self.lr = lr
        self.key_added = key_added
        self.gradient_clipping = gradient_clipping
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.random_seed = random_seed
        self.save_reconstruction = save_reconstruction
        self.device = device
        self.save_path = save_path
        self.ncluster = ncluster
        self.repeat = repeat
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.temperature = temperature

        seed_everything(self.random_seed)#固定种子
        self.adata.X = sp.csr_matrix(self.adata.X)
        if 'highly_variable' in self.adata.var.columns:
            self.adata_Vars = self.adata[:, self.adata.var['highly_variable']]
        else:
            self.adata_Vars = self.adata

        if verbose:
            print('Size of Input: ', self.adata_Vars.shape)
        if 'Spatial_Net' not in adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
        start_time = time.time()
        self.data, self.img = Transfer_img_Data(self.adata_Vars)  # 将基因表达数据及空间网络信息（包括2个cell对应的索引），图像数据及空间网络信息转换为PyTorch中的Data对象
        Transfer_img_Data_time = time.time() - start_time
        print("Transfer_img_Data time:", Transfer_img_Data_time)

        # data.x = torch.cat([data.x, img.x], dim=1)  # 将表达信息与图像信息合并
        # my_dataset = MyDataset(data, patchs1, centers1)
        # loader = DataLoader(my_dataset, batch_size=1, sampler=SequentialSampler(my_dataset), collate_fn=my_collate_fn)
        # loader = DataLoader(data, batch_size=1, shuffle=True)
        # model = TransImg(hidden_dims=[gene_dim, img_dim] + hidden_dims).to(device)  # 加载模型
        self.num_nodes = self.data.x.size(0)
        self.noramalized_adj,self.adj = preprocess_adj(self.data.edge_index, self.num_nodes)  # convert edge_index to adj,then Laplacian normalization
        # bgene = data.x[:, :gene_dim]
        # gene_weight = 0.2
        # img_weight = 0.4
        # combine_weight = 0.4
        # cl_weight = 0.1
        # num_nodes = bgene.size(0)

    def train(self):
        self.gene_dim = self.data.x.shape[1]
        self.img_dim = self.img.x.shape[1]
        self.model = TransImg_CosMx(hidden_dims=[self.gene_dim, self.img_dim] + self.hidden_dims, device=self.device)
        torch.save(self.model.state_dict(), os.path.join(self.save_path, 'init.pth'))
        print(self.device)
        self.data = self.data.to(self.device)
        self.img = self.img.to(self.device)
        # self.patchs = pd.DataFrame(self.patchs).values
        # # patchs1 = np.stack(patchs1)
        # self.patchs = torch.tensor(self.patchs)
        # self.patchs = self.patchs.reshape(-1, 3, 120, 120)
        # self.patchs = self.patchs.to(self.device)
        # self.centers = pd.DataFrame(self.centers).values
        # self.centers = torch.tensor(self.centers)
        # self.centers = self.centers.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Total_Training_time = 0
        gloss_list = []
        iloss_list = []
        closs_list = []
        clloss_list = []
        mask_loss_list=[]
        for epoch in tqdm(range(1, self.n_epochs + 1)):
            self.model.train()
            self.optimizer.zero_grad()
            print(torch.cuda.is_available())
            start_time = time.time()#,gz_contra, iz_contra 
            gz, iz, cz,gout, iout,cout,gz_contra, iz_contra,x_init,x_rec = self.model(self.data.x, self.img.x, self.noramalized_adj,self.data.edge_index, self.temperature,device=self.device)#self.patchs, self.centers,
            training_time = time.time() - start_time
            print("Training time:", training_time)
            gloss = F.mse_loss(self.data.x, gout)
            mask_loss = sce_loss(x_rec, x_init)
            iloss = F.mse_loss(self.data.x, iout)
            # iloss = F.mse_loss(self.img.x, iout)
            closs = F.mse_loss(self.data.x, cout)
            cl_loss =CL_loss_with_Spatial_graph(gz_contra, iz_contra,self.adj,self.temperature) #CL_loss_SCprotein
            loss = gloss * self.lambda_1+ iloss * self.lambda_2+ closs * self.lambda_3 + cl_loss * self.lambda_4+mask_loss*5#
            print("gloss:", gloss * self.lambda_1)
            print("iloss:", iloss * self.lambda_2)
            print("closs:", closs * self.lambda_3)
            print("cl_loss:",cl_loss * self.lambda_4)
            print("mask_loss:", mask_loss*5)
            gloss_list.append(gloss.item())
            iloss_list.append(iloss.item())
            closs_list.append(closs.item())
            clloss_list.append(cl_loss.item())
            mask_loss_list.append(mask_loss.item())
            print("loss:", loss)
            loss.backward()
            self.optimizer.step()
            Total_Training_time += training_time
            if epoch ==900:
                # self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'final_%d.pth' % (self.repeat))))
                print("Total Training time:", Total_Training_time)
                with torch.no_grad():
                    # self.adata.X = sp.csr_matrix(self.adata.X)
                    # # adata.X = adata.X.A
                    # if 'highly_variable' in self.adata.var.columns:
                    #     self.adata_Vars = self.adata[:, self.adata.var['highly_variable']]
                    # else:
                    #     self.adata_Vars = self.adata
                    # print('Size of Input: ', self.adata_Vars.shape)
                    # # if 'Spatial_Net' not in adata.uns.keys():
                    # #     raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")
                    # self.data, self.img = Transfer_img_Data(self.adata_Vars)
                    # self.model = TransImg(hidden_dims=[self.gene_dim, self.img_dim] + self.hidden_dims,
                    #                       device=self.device)
                    # # model_path = os.path.join(save_path, pretrain)
                    # # print(model_path)
                    # # model.load_state_dict(torch.load(model_path))
                    # self.data = self.data.to(self.device)
                    # self.img = self.img.to(self.device)
                    self.model.eval()#,gz_contra, iz_contra
                    gz, iz, cz,gout, iout,cout,gz_contra, iz_contra,x_init,x_rec= self.model(self.data.x, self.img.x, self.noramalized_adj,self.data.edge_index, self.temperature,device=self.device)#self.patchs, self.centers,
                    # # self.adata_Vars.obsm['pred'] = cz.clone().detach().cpu().numpy()
                    # self.adata_Vars.obsm['pred'] =  F.normalize(cz, p=2, dim=1).clone().detach().cpu().numpy()
                    # cell_shape = self.adata_Vars.shape[0]
                    # sc.pp.neighbors(self.adata_Vars, use_rep='pred')
                    # sc.tl.umap(self.adata_Vars)
                    # plt.rcParams["figure.figsize"] = (3, 3)
                    # sc.settings.figdir = self.save_path
                    # ax = sc.pl.umap(self.adata_Vars, color=['Ground Truth'], show=False, title='combined latent variables')
                    # plt.savefig(os.path.join(self.save_path, 'umap_final.pdf'), bbox_inches='tight')

                    # print(adata_Vars.obsm['pred'])
                    self.adata_Vars.obsm['pred'] = F.normalize(cout, p=2, dim=1).detach().cpu().numpy().astype(np.float32)  # do cz normalization
                    output = cout.detach().cpu().numpy().astype(np.float32)
                    # output[output < 0] = 0  ########delete
                    self.adata_Vars.layers['recon'] = output
                    #self.adata.obsm['imgs'] = self.adata.obsm['imgs'].to_numpy()
        # # plt.close('all')
        # plt.plot(gloss_list, label='gloss')
        #plt.plot(range(1, len(gloss_list) + 1), gloss_list, label='gloss')
        #plt.xlabel('Epoch')
        #plt.ylabel('Loss')
        #plt.title('gloss curve')
        #plt.legend()
        #plt.savefig(os.path.join(self.save_path, 'gloss.png'))  # 保存图片
        #plt.close()

        # plt.plot(iloss_list, label='iloss')
        #plt.plot(range(1, len(iloss_list) + 1), iloss_list, label='iloss')
        #plt.xlabel('Epoch')
        #plt.ylabel('Loss')
        #plt.title('iloss curve')
        #plt.legend()
        #plt.savefig(os.path.join(self.save_path, 'iloss.png'))  # 保存图片
        #plt.close()

        # plt.plot(closs_list, label='closs')
        #plt.plot(range(1, len(closs_list) + 1), closs_list, label='closs')
        #plt.xlabel('Epoch')
        #plt.ylabel('Loss')
        #plt.title('closs curve')
        #plt.legend()
        #plt.savefig(os.path.join(self.save_path, 'closs.png'))  # 保存图片
        #plt.close()

        # plt.plot(clloss_list, label='cl_loss')
        #plt.plot(range(1, len(clloss_list) + 1), clloss_list, label='cl_loss')
        #plt.xlabel('Epoch')
        #plt.ylabel('Loss')
        #plt.title('cl_loss curve')
        #plt.legend()
        #plt.savefig(os.path.join(self.save_path, 'cl_loss.png'))  # 保存图片
        #plt.close()

        #plt.plot(range(1, len(mask_loss_list) + 1), clloss_list, label='mask_loss')
        #plt.xlabel('Epoch')
        #plt.ylabel('Loss')
        #plt.title('mask_loss curve')
        #plt.legend()
        #plt.savefig(os.path.join(self.save_path, 'mask_loss.png'))  # 保存图片
        #plt.close()

        # # 训练结束后绘制损失函数曲线并保存为图片
        # plt.plot(cl_loss_history, label='CL Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('CL Loss')
        # plt.title('CL Loss Curve')
        # plt.legend()
        # plt.savefig(os.path.join(self.save_path, 'cl_loss_curve.png'))  # 将损失函数曲线保存为图片
        # plt.close()  # 关闭绘图窗口，释放资源
        return self.adata_Vars