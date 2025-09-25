import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, gene_embedding_dim, image_embedding_dim, temperature=1.0, projection_dim=128,device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.temperature = temperature
        self.device=device
        self.gene_projection = nn.Linear(gene_embedding_dim, projection_dim).to(self.device)
        self.image_projection = nn.Linear(image_embedding_dim, projection_dim).to(self.device)
        #self.loss_weight = nn.Parameter(torch.Tensor([1.0])).to(device) # Add a weight parameter

    def forward(self, gene_features, image_features):
        # Projection to feature space
        gene_embeddings = self.gene_projection(gene_features)
        image_embeddings = self.image_projection(image_features)
        gene_embeddings = F.normalize(gene_embeddings, dim=-1, p=2)
        image_embeddings = F.normalize(image_embeddings, dim=-1, p=2)

        # Calculate similarity scores
        similarity_gene_to_image = torch.matmul(gene_embeddings, image_embeddings.t()) / self.temperature
        similarity_image_to_gene = torch.matmul(image_embeddings, gene_embeddings.t()) / self.temperature

        # Contrastive learning loss calculation
        loss_gene_to_image = self.contrastive_loss(similarity_gene_to_image)
        loss_image_to_gene = self.contrastive_loss(similarity_image_to_gene)

        # Total loss
        total_loss = (loss_gene_to_image + loss_image_to_gene) / 2.0
        return total_loss.mean()

    def contrastive_loss(self, similarity_scores):
        batch_size = similarity_scores.size(0)
        mask = torch.eye(batch_size, device=similarity_scores.device)
        positive_samples = torch.diag(similarity_scores)
        negative_samples = (1 - mask) * similarity_scores
        numerator = torch.exp(positive_samples / self.temperature)
        denominator = numerator + torch.sum(torch.exp(negative_samples / self.temperature), dim=-1)
        loss = -torch.log(numerator / denominator)
        #weighted_loss = loss * self.loss_weight  # Apply the weight
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