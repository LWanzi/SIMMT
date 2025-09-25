import pandas as pd
from torch_geometric.nn import  LayerNorm, GATConv, GCNConv#TransformerConv,
import torch.nn.functional as F
import torch
import math
import torch.nn as nn
import numpy as np
# torch.set_default_tensor_type(torch.DoubleTensor)
# from transformer import ViT
# from GATLayer import MultiHeadGAT
import time
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from itertools import repeat
import collections.abc
from functools import partial
from preprocess import adj_to_edge_index
# from Transformerconv import MyTransformerConv
# from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.nn.functional import normalize
class GCN(nn.Module):#GCN
    def __init__(self, in_dim, out_dim,apply_activation=True):
        super(GCN, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.norm = nn.BatchNorm1d(out_dim)
        self.activate = nn.ReLU()
        self.apply_activation = apply_activation
        #self.device = device
        # 将参数张量移动到指定设备上
        #self.weight = self.weight.to(self.device)
    def reset_parameters(self):# Initialize weights using Xavier initialization
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, adj, x):
        #self.weight = nn.Parameter(self.weight.to(x.device))
        z = torch.matmul(x, self.weight)
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  ##for linux gpu
        adj = adj.to(x.device)  ###for gpu
        #print(adj.device)
        #print(z.device)
        #z = torch.matmul(adj, z)
        z=torch.spmm(adj, z)
        #print(z.device)
        #self.norm = self.norm.to(x.device)
        #self.activate= self.activate.to(x.device)
        if self.apply_activation:#是否进行激活函数及规范化
            h = self.activate(self.norm(z))
            # h = self.activate(z)
        else:
            h = z
        return h
class TransformerConv(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 beta=False, dropout=0., edge_dim=None, bias=True,
                 root_weight=True):
        super(TransformerConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        if concat:
            self.lin_skip = nn.Linear(in_channels, heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = None
        else:
            self.lin_skip = nn.Linear(in_channels, out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.lin_skip.weight)
        if self.lin_beta is not None:
            nn.init.xavier_uniform_(self.lin_beta.weight)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        H, C = self.heads, self.out_channels

        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        if self.lin_edge is not None:
            edge_attr = self.lin_edge(edge_attr).view(-1, H, C)
            key = key + edge_attr

        alpha = (query * key).sum(dim=-1) / (C ** 0.5)
        alpha = F.softmax(alpha, dim=-1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value * alpha.view(-1, H, 1)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x)
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = torch.sigmoid(beta)
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            if return_attention_weights:
                return out, alpha
        return out
class MLPBlock(torch.nn.Module):#MLP
    def __init__(self, in_channels, out_channels):
        super(MLPBlock, self).__init__()
        self.fc = torch.nn.Linear(in_channels, out_channels)
        self.relu=torch.nn.ReLU()

    def forward(self, x):
        # return self.relu(self.fc(x))
        return self.fc(x)

    def reset_parameters(self):  # Initialize weights using Xavier initialization
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)
class MyTransformerConv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 beta=False, dropout=0., edge_dim=None, bias=True,
                 root_weight=True, **kwargs):
        super(MyTransformerConv, self).__init__('add', **kwargs)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.dropout = dropout
        self.edge_dim = edge_dim

        self.lin_key = nn.Linear(in_channels[0], heads * out_channels)
        self.lin_query = nn.Linear(in_channels[1], heads * out_channels)
        self.lin_value = nn.Linear(in_channels[0], heads * out_channels)

        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        if concat:
            self.lin_skip = nn.Linear(in_channels[1], heads * out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = None
        else:
            self.lin_skip = nn.Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = nn.Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.lin_skip.weight)
        if self.lin_beta is not None:
            nn.init.xavier_uniform_(self.lin_beta.weight)

    def forward(self, x, edge_index, edge_attr=None, return_attention_weights=None):
        H, C = self.heads, self.out_channels

        if isinstance(x, torch.Tensor):
            x = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)
        min_node_index = edge_index.min().item()
        max_node_index = edge_index.max().item()

        print("最小节点索引:", min_node_index)
        print("最大节点索引:", max_node_index)
        out = self.propagate(edge_index, query=query, key=key, value=value,
                             edge_attr=edge_attr)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out = out + x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, torch.Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        if self.lin_edge is not None:
            assert edge_attr is not None
            edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            key_j = key_j + edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        if edge_attr is not None:
            out = out + edge_attr

        out = out * alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')

class TransformerConv1(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.):
        super(TransformerConv1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        self.weight = nn.Linear(in_channels, out_channels * heads, bias=False)
        self.att = nn.Linear(out_channels, heads, bias=False)
        self.lin = nn.Linear(in_channels * 2, out_channels)

        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.weight)
        nn.init.xavier_uniform_(self.att.weight)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index):
        x = self.weight(x).view(-1, self.heads, self.out_channels)
        x = F.dropout(x, p=self.dropout, training=self.training)

        alpha = (x * self.att(x).view(-1, self.heads, 1)).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        print(alpha.shape)
        print(edge_index[0].shape)
        print(edge_index[1].shape)
        alpha = SparseTensor(row=edge_index[0], col=edge_index[1], value = alpha[edge_index[0]],#value=alpha
                             sparse_sizes=(x.size(0), x.size(0)))
        alpha = alpha.to_dense()
        alpha = torch.softmax(alpha, dim=1)
        # alpha = alpha.softmax(dim=1)
        alpha = torch.sparse_coo_tensor(alpha.nonzero().t(),
                                        alpha[alpha.nonzero().t()[0], alpha.nonzero().t()[1]],
                                        alpha.size())
        # alpha = alpha.to_sparse_csr()
        out = matmul(alpha, x)

        out = out.view(-1, self.heads * self.out_channels)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.elu(self.lin(out))

        return out

class ViTEncoding_CosMx(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=4,learning_rate=1e-3, dropout=0.2, n_pos=5413,# 128  5457
                 heads=16, patch_size=120,n_genes=980):#patch_size=50,n_genes=7500 for DLPFC    43200   patch_size=120,n_genes=980   heads=[2,1]
        super(ViTEncoding_CosMx,self).__init__()
        patch_dim=3*patch_size*patch_size

        self.patch_embedding=nn.Linear(patch_dim,out_dim)
        # Vision Transformer Layer
        self.x_embed = nn.Embedding(n_pos, out_dim)
        self.y_embed = nn.Embedding(n_pos, out_dim)
        self.vit = ViT(dim=out_dim, depth=n_layers, heads=heads, mlp_dim=2 * out_dim, dropout=dropout,#heads=heads[0]
                       emb_dropout=dropout)
        # self.gat=MultiHeadGAT(in_features=out_dim,nhid=1024,out_features=32,heads=heads[1],dropout=dropout,alpha=0.01)
        self.gene_head=nn.Sequential(nn.LayerNorm(out_dim),nn.Linear(out_dim,n_genes))
        # self.gene_head = nn.Sequential(
        #    nn.Linear(32, 2048),
        #    nn.ReLU(),
        #    nn.LayerNorm(2048),
        #    nn.Linear(2048, n_genes)
        # )
    def forward(self, patchs1,centers,adj):
        N,C,H,W=patchs1.shape
        patchs1=patchs1.reshape(-1,C*H*W)
        patchs1 = patchs1.unsqueeze(0)#在第0位添加维度，此时为1，spot,3*2h*2w
        patchs1 = self.patch_embedding(patchs1)
        centers=centers.unsqueeze(0)####for single fov use
        centers=centers.long()
        centers_x1 = self.x_embed(centers[:, :, 0])
        centers_y1 = self.y_embed(centers[:, :, 1])#.squeeze(0) delete one dim
        x=patchs1+centers_x1+centers_y1

        # Vision Transformer Encoding
        h = self.vit(x)#viT encoding
        h=h.reshape(h.shape[1],-1)#convert three dim into two dim 4221 32
        # h=self.gat(h,adj)#4221 32
        # print(h.shape)
        x1=self.gene_head(h)#decoding

        return h,x1


class TransImg_CosMx(torch.nn.Module):
    def __init__(self, hidden_dims, use_img_loss=False,device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        # [in_dim, img_dim, num_hidden, out_dim] = hidden_dims
        [in_dim, img_dim, out_dim] = hidden_dims
        # self.GCN1 = GCN(in_dim, num_hidden,apply_activation=False).to(device)
        # self.GCN2 = GCN(num_hidden, out_dim,apply_activation=False).to(device)#apply_activation=False不进行规范化
        # self.GCN3 = GCN(out_dim, num_hidden,apply_activation=False).to(device)
        # self.GCN4 = GCN(num_hidden, in_dim,apply_activation=False).to(device)
        # self.GCN1 = TransformerConv(in_dim, num_hidden).to(device)
        # self.GCN2 = TransformerConv(num_hidden, out_dim).to(device)#apply_activation=False不进行规范化
        # self.GCN3 = TransformerConv(out_dim, num_hidden).to(device)
        # self.GCN4 = TransformerConv(num_hidden, in_dim).to(device)
        self.GCN1 = TransformerConv(in_dim, out_dim).to(device)
        self.GCN2 = TransformerConv(out_dim, in_dim).to(device)
        self.device=device
        self.mlp_hid = 512
        self.mlp_out = 128
        #######
        # self.viT_encoder=ViTEncoding_CosMx(img_dim, out_dim).to(device)
        # self.iGCN1 = TransformerConv(img_dim, num_hidden).to(device)
        # self.iGCN2 = TransformerConv(num_hidden, out_dim).to(device)#apply_activation=False不进行规范化
        # self.iGCN3 = TransformerConv(out_dim, num_hidden).to(device)
        # self.iGCN4 = TransformerConv(num_hidden, in_dim).to(device)
        self.iGCN1 = TransformerConv(img_dim, out_dim).to(device)
        self.iGCN2 = TransformerConv(out_dim, in_dim).to(device)#in_dim  img_dim
        #######
        # self.mlp1 = MLPBlock(out_dim * 2, in_dim).to(device)
        # self.combine_GCN1 = GCN(out_dim * 2, out_dim,apply_activation=False).to(device)
        # # # self.neck = GCN(num_hidden+out_dim, out_dim, apply_activation=False)
        # self.combine_GCN2 = GCN(out_dim, out_dim,apply_activation=False).to(device)
        # self.combine_GCN3 = GCN(out_dim,num_hidden,apply_activation=False).to(device)
        # self.combine_GCN4 = GCN(num_hidden, in_dim,apply_activation=False).to(device)
        #self.combine_GCN1 = TransformerConv(out_dim * 2, out_dim).to(device)
        #self.combine_GCN2 = TransformerConv(out_dim, out_dim).to(device)
        #self.combine_GCN3 = TransformerConv(out_dim,num_hidden).to(device)
        #self.combine_GCN4 = TransformerConv(num_hidden, in_dim).to(device)

        #######
        self.gene_proj = Mlp(out_dim, self.mlp_hid, self.mlp_out).to(device)
        self.image_proj = Mlp(out_dim, self.mlp_hid, self.mlp_out).to(device)

        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=out_dim * 2, nhead=1,  # 256
                                                                  dim_feedforward=256).to(device)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1).to(device)
        self.Common_view = nn.Sequential(
            nn.Linear(out_dim * 2, in_dim),  # or 128  in_dim
        ).to(device)
        # layernorm
        # self.norm1 = LayerNorm(num_hidden)
        self.norm2 = LayerNorm(out_dim)
        # relu
        self.activate = F.elu

        self.dropout = 0.0  # Set the dropout value as needed
        self._mask_rate = 0.8#0.8
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim)).to(device)

        self.reset_parameters()
    def reset_parameters(self):
        self.GCN1.reset_parameters()
        self.GCN2.reset_parameters()
        # self.GCN3.reset_parameters()
        # self.GCN4.reset_parameters()

        self.iGCN1.reset_parameters()
        self.iGCN2.reset_parameters()
        # self.iGCN3.reset_parameters()
        # self.iGCN4.reset_parameters()

        # self.mlp1.reset_parameters()

        # self.combine_GCN1.reset_parameters()
        # self.combine_GCN2.reset_parameters()
        # self.combine_GCN3.reset_parameters()
        # self.combine_GCN4.reset_parameters()

        self.gene_proj.reset_parameters()
        self.image_proj.reset_parameters()

    def encoding_mask_noise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)  # 随机打乱spot排列 (spots,)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)  # 掩码的spot数量
        mask_nodes = perm[: num_mask_nodes]  # 掩码的spot索引 (num_mask_nodes,)
        keep_nodes = perm[num_mask_nodes:]  # 保留的spot索引 (total_nodes-num_mask_nodes,)

        out_x = x.clone()
        token_nodes = mask_nodes
        # out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token  # 根据掩码的spot索引，将相应位置的基因表达值全设为0，enc_mask_token为全0向量

        return out_x, (mask_nodes, keep_nodes)  # (spots,spots),(spots,200),((num_mask_nodes,),(total_nodes-num_mask_nodes,))
    # def forward(self, features, img_feat, edge_index):
    def forward(self, features, img_feat, adj,edge_index,temperature,device):#edge_index   patchs1,centers,
        #edge_index=adj_to_edge_index(adj)
        #edge_index = torch.tensor(edge_index)
        #print(edge_index.shape)
        z = F.dropout(features, self.dropout, self.training)
        # num_nodes = features.size(0)
        # adj = preprocess_adj(edge_index, num_nodes)#convert edge_index to adj,then Laplacian normalization
        start_time = time.time()
        #z=features
        #g1 = self.GCN1(adj, z)
        #g2 = self.GCN2(adj, g1)
        #g3 = self.GCN3(adj, g2)
        #g4 = self.GCN4(adj, g3)
        # adj=adj.to(device)
        #edge_index=adj_to_edge_index(adj)
        #print(edge_index.shape)
        # g1 = self.GCN1(z,edge_index)
        # g2 = self.GCN2(g1,edge_index)
        # g3 = self.GCN3(g2,edge_index)
        # g4 = self.GCN4(g3,edge_index)
        # g1 = self.activate(self.GCN1(z,edge_index))
        # g2 = self.GCN2(g1, edge_index)
        # g3 = self.activate(self.GCN3(g2, edge_index))
        # g4 = self.GCN4(g3, edge_index)
        #gene expression data mask
        masked_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(features, self._mask_rate)

        g1 = self.GCN1(masked_x, edge_index)#z
        g2 = self.GCN2(g1, edge_index)

        x_init = features[mask_nodes]  # 掩码部分之前的值 （掩码细胞数量，gene dim）
        x_rec = g2[mask_nodes]# 掩码部分重构后的值（掩码细胞数量，gene dim）

        graph_conv_time = time.time() - start_time
        print("GCN time:", graph_conv_time)
        # ViTEncoding
        start_time = time.time()
        ######
        # i1 = self.activate(self.iGCN1(img_feat,edge_index))
        # i2 = self.iGCN2(i1, edge_index)
        # i3 = self.activate(self.iGCN3(i2, edge_index))
        # i4 = self.iGCN4(i3, edge_index)
        i1 = self.iGCN1(img_feat, edge_index)
        i2 = self.iGCN2(i1, edge_index)
        #i2,i4 = self.viT_encoder(patchs1,centers,adj)#img2--encoded representation, img4--reconstructed x
        vit_encoding_time = time.time() - start_time
        print("ViTEncoding time:", vit_encoding_time)
        #######
        start_time = time.time()
        # concat = torch.cat([g2, i2], dim=1)#h2(4381,30) img2(4381,512)
        # c4 = self.mlp1(concat)
        concat = torch.cat([g1, i1], dim=1)  # h2(4381,30) img2(4381,512)
        #######, S
        c2, S = self.TransformerEncoderLayer(concat)  # 用transformer编码后的潜在表示Z^：256，1024   全局结构关系矩阵S：256，256
        c4 = normalize(self.Common_view(c2), dim=1)  # mlp输出 256,128


        #c1 = self.activate(self.combine_GCN1(adj,concat))
        #c2 = self.combine_GCN2(adj,c1)  # 组合潜在表示
        #c3 = self.activate(self.combine_GCN3(adj,c2))
        #c4 = self.combine_GCN4(adj,c3)  # 组合重构输出

        #c1 = self.activate(self.combine_GCN1(concat,edge_index))
        #c2 = self.combine_GCN2(c1,edge_index)  # 组合潜在表示
        #c3 = self.activate(self.combine_GCN3(c2,edge_index))
        #c4 = self.combine_GCN4(c3,edge_index)  # 组合重构输出
        combine_GCN_time = time.time() - start_time
        print("Combine GCN time:", combine_GCN_time)
        gz_contra = self.gene_proj(g1)
        iz_contra = self.image_proj(i1)
        # cl_loss = self.cl(gz, iz)
        # return g2,i2,concat,g4,i4,c4,gz_contra,iz_contra
        # return g2, i2, c2, g4, i4, c4, gz_contra, iz_contra
        return g1, i1, c2, g2, i2, c4, gz_contra, iz_contra,x_init,x_rec

class MultiHeadGAT(nn.Module):
    def __init__(self, in_features, nhid, out_features, dropout, alpha, heads=4):
        super(MultiHeadGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(in_features, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * heads, out_features, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adj = adj.to(device)
        x = torch.cat([att(x, adj).squeeze(0) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.01, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    # @get_local('attn')
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        # print(attn.shape)
        # quit()
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()

    def forward(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_latent(x)
        return x




def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse


to_2tuple = _ntuple(2)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.ReLU,#GELU
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()
        out_features = out_features or in_features#如果out_features参数未指定（为 None），则将其设置为in_features的值
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)#将bias参数转换为一个元组类型
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.device=device
        self.to(self.device)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop1(x)
        # x = self.norm(x)
        x = self.fc2(x)
        # x = self.drop2(x)
        return x

    def reset_parameters(self):
        # 初始化模型的权重和偏置
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)