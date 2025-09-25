import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
class ContrastiveLoss(nn.Module):
    def __init__(self, gene_embedding_dim, image_embedding_dim, temperature=1.0, projection_dim=128,device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
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
        # print("Image embeddings device:", image_embeddings.device)
        # Calculating the Loss
        logits = (image_embeddings @ gene_embeddings.T) / self.temperature
        genes_similarity = gene_embeddings @ gene_embeddings.T
        images_similarity = image_embeddings @ image_embeddings.T
        targets = F.softmax(
            (genes_similarity + images_similarity) / 2 * self.temperature, dim=-1
        )
        images_loss = cross_entropy(logits, targets, reduction='none')
        genes_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss=genes_loss
        #loss = (genes_loss + images_loss) / 2.0  # shape: (batch_size)
        return loss.mean()

# # 定义gene特征维度和image特征维度
# gene_embedding_dim = 512  # gene特征维度
# image_embedding_dim = 512   # image特征维度
#
# # 创建 ContrastiveLoss 对象
# CL_loss = ContrastiveLoss(gene_embedding_dim, image_embedding_dim, temperature=0.07, projection_dim=128)
#
# # 计算对比学习损失
# cl_loss = CL_loss(gz, iz)