import torch
import torch.nn as nn
import torch.nn.functional as F
def CL_loss(gene_embeddings, image_embeddings,temperature=1):
    gene_embeddings = F.normalize(gene_embeddings, dim=-1, p=2)
    image_embeddings = F.normalize(image_embeddings, dim=-1, p=2)
    # Calculate similarity scores
    similarity_gene_to_image = torch.matmul(gene_embeddings, image_embeddings.t()) / temperature
    similarity_image_to_gene = torch.matmul(image_embeddings, gene_embeddings.t()) / temperature
    # Contrastive learning loss calculation
    loss_gene_to_image = loss(similarity_gene_to_image)
    loss_image_to_gene = loss(similarity_image_to_gene)
    # Total loss
    total_loss = (loss_gene_to_image + loss_image_to_gene) / 2.0
    return total_loss.mean()

def loss(similarity_scores,temperature=1.0):
    batch_size = similarity_scores.size(0)
    mask = torch.eye(batch_size, device=similarity_scores.device)
    positive_samples = torch.diag(similarity_scores)
    negative_samples = (1 - mask) * similarity_scores
    numerator = torch.exp(positive_samples / temperature)
    denominator = numerator + torch.sum(torch.exp(negative_samples / temperature), dim=-1)
    loss = -torch.log(numerator / denominator)
    return loss.mean()
def CL_loss1(gene_embeddings, image_embeddings,temperature=1.0):
    gene_embeddings = F.normalize(gene_embeddings, dim=-1, p=2)
    image_embeddings = F.normalize(image_embeddings, dim=-1, p=2)
    logits = temperature * gene_embeddings @ image_embeddings.T
    labels = torch.arange(logits.shape[0])
    #print(logits.device)
    labels=labels.to(logits.device)
    #accuracy = (logits.argmax(dim=1) == labels).float().mean()
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss

def sim(z1: torch.Tensor, z2: torch.Tensor):#x余弦相似性
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())
def semi_loss(z1: torch.Tensor, z2: torch.Tensor,temperature=0.5):
    f = lambda x: torch.exp(x /temperature)#x作为输入，返回exp(x / self.tau)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))
    return -torch.log(
        between_sim.diag()
        / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))#.sum(1)按行求和
def CL_loss_SCprotein(h1, h2,temperature=0.5):#h1 gene嵌入   h2 image嵌入
    # h1 = self.projection(z1)
    # h2 = self.projection(z2)

    l1 = semi_loss(h1, h2,temperature)
    l2 = semi_loss(h2, h1,temperature)

    ret = (l1 + l2) * 0.5
    ret = ret.mean()#sum()

    return ret
    # Calculating the Loss



def CL_loss_BLEEP(h1, h2,temperature=1):#h1 gene嵌入   h2 image嵌入
    logits = (h1@h2.T) / temperature
    images_similarity = h2@h2.T
    spots_similarity = h1@h1.T
    targets = F.softmax(
        ((images_similarity + spots_similarity) / 2) / temperature, dim=-1
    )
    spots_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss = (images_loss + spots_loss) / 2.0  # shape: (batch_size)
    return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def sce_loss(x, y, alpha=3):#3
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss