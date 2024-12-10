import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        # print("Gene embeddings device:", gene_embeddings.device)
        gene_embeddings = F.normalize(gene_embeddings, dim=-1, p=2)
        image_embeddings = F.normalize(image_embeddings, dim=-1, p=2)

        # Calculate similarity scores (dot product)计算相似度分数（内积）
        similarity_gene_to_image = torch.matmul(gene_embeddings, image_embeddings.t()) / self.temperature
        similarity_image_to_gene = torch.matmul(image_embeddings, gene_embeddings.t()) / self.temperature

        # Contrastive learning loss calculation
        loss_gene_to_image = self.contrastive_loss(similarity_gene_to_image)
        # loss_image_to_gene = self.contrastive_loss(similarity_image_to_gene)

        # Total loss
        # total_loss = (loss_gene_to_image + loss_image_to_gene) / 2.0
        total_loss = loss_gene_to_image
        return total_loss.mean()

    def contrastive_loss(self, similarity_scores):
        batch_size = similarity_scores.size(0)
        # Diagonal for positive samples, rest for negative samples对角线为正样本，其余为负样本
        mask = torch.eye(batch_size, device=similarity_scores.device)#对角线上的元素为1，其余为0
        positive_samples = torch.diag(similarity_scores)#提取正样本相似度得分
        negative_samples = (1 - mask) * similarity_scores
        # Calculate contrastive loss
        # loss = -torch.log(
            # torch.exp(positive_samples / self.temperature) / torch.sum(torch.exp(negative_samples / self.temperature), dim=-1))
        numerator = torch.exp(positive_samples / self.temperature)
        denominator = numerator + torch.sum(torch.exp(negative_samples / self.temperature), dim=-1)
        loss = -torch.log(numerator / denominator)
        return loss.mean()

# Example usage:
# criterion = ContrastiveLoss(gene_embedding_dim, image_embedding_dim)
# loss = criterion(gene_features, image_features)

# # 定义gene特征维度和image特征维度
# gene_embedding_dim = 512  # gene特征维度
# image_embedding_dim = 512   # image特征维度
#
# # 创建 ContrastiveLoss 对象
# CL_loss = ContrastiveLoss(gene_embedding_dim, image_embedding_dim, temperature=0.07, projection_dim=128)
#
# # 计算对比学习损失
# cl_loss = CL_loss(gz, iz)