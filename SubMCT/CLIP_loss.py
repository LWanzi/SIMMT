import torch
import torch.nn as nn
import torch.nn.functional as F
class ContrastiveLoss(nn.Module):
    def __init__(self, gene_embedding_dim, image_embedding_dim, temperature=1.0, projection_dim=128,device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
):
        super().__init__()
        self.temperature = temperature
        self.device=device
        self.gene_projection = nn.Linear(gene_embedding_dim, projection_dim).to(self.device)
        self.image_projection = nn.Linear(image_embedding_dim, projection_dim).to(self.device)

    def forward(self, gene_features, image_features):
        # Projection to feature space
        gene_embeddings = self.gene_projection(gene_features)
        image_embeddings = self.image_projection(image_features)

        # Calculating the dot product similarity:predict
        logits = torch.matmul(gene_embeddings, image_embeddings.T) / self.temperature

        # Calculating the targets:consider as ground truth
        similarity_matrix = torch.matmul(
            F.normalize(gene_embeddings, p=2, dim=-1),
            F.normalize(image_embeddings, p=2, dim=-1).T
        )
        targets = F.softmax(similarity_matrix * self.temperature, dim=-1)
        # 获取 logits 的大小并转移到 GPU
        logits_size = logits.size(0)
        # 创建对应的索引张量并转移到 GPU
        target = torch.arange(logits_size).to(self.device)
        # Calculating the contrastive loss
        loss = F.cross_entropy(logits.to(self.device), targets, reduction='mean')
        return loss
# 定义gene特征维度和image特征维度
# gene_embedding_dim = 512  # gene特征维度
# image_embedding_dim = 512   # image特征维度
#
# # 创建 ContrastiveLoss 对象
# CL_loss = ContrastiveLoss(gene_embedding_dim, image_embedding_dim, temperature=0.07, projection_dim=128)
#
# # 计算对比学习损失
# cl_loss = CL_loss(gz, iz)