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


def semi_loss_with_neighbors(z1: torch.Tensor, z2: torch.Tensor, A: torch.Tensor, temperature=0.5):
    """考虑邻居信息的对比损失函数"""
    # 计算z1和z2之间的相似性以及z1和z1之间的相似性
    z1_sim_z1 = sim(z1, z1)  # intra-modal similarity for z1
    z1_sim_z2 = sim(z1, z2)  # cross-modal similarity

    # 邻居信息的应用:
    # 将A broadcast到相似性的维度，mask掉非邻居相似性
    # 将A转换为与相似性矩阵相同维度，非邻居的地方设置为0（即相似性不被计算）
    # 但是由于在损失函数中，我们需要进行sum操作，这里我们需要mask掉非邻居的相似性，即将它们忽略
    # 因此，我们需要将相似性矩阵乘以邻接矩阵
    # 注意邻接矩阵A可能需要转换为浮点或其他类型以进行矩阵乘法

    # 1. 处理邻接矩阵A
    n = A.size(0)
    # 确保A的大小与相似性矩阵一致
    if len(z1_sim_z1.shape) == 2:
        assert z1_sim_z1.shape == (n, n), "相似性矩阵的维度应与邻接矩阵相同"
    A = A.to(z1.device)

    # 2. 计算交叉模态的邻居相似性
    # 对于每个样本i，其邻居j在 modality2中的相似性
    # 使用A来mask掉非邻居的信息
    cross_modal_neighbor_sim = torch.mul(z1_sim_z2, A)  # element-wise multiplication

    # 3. 计算总的相似性包括自己和邻居
    total_similarity = torch.mul(z1_sim_z1, torch.ones_like(A)) + cross_modal_neighbor_sim

    # 4. 应用温度系数
    f = lambda x: torch.exp(x / temperature)
    total_similarity = f(total_similarity)

    # 5. 计算分母
    denominator = total_similarity.sum(dim=1)

    # 6. 计算负对数损失
    # 分子是样本i与跨模态对应的相似性
    numerator = f(z1_sim_z2)
    numerator = numerator.diag()

    # 计算损失
    loss = -torch.log(numerator / denominator)

    return loss.mean()
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


def semi_loss_with_neighbors1(z1: torch.Tensor, z2: torch.Tensor, A: torch.Tensor, temperature=0.5):
    """考虑邻居信息的对比损失函数"""
    # 计算同一模态的相似性矩阵
    refl_sim = sim(z1, z1)
    # 计算跨模态的相似性矩阵
    between_sim = sim(z1, z2)

    # 应用温度系数
    f = lambda x: torch.exp(x / temperature)
    refl_sim = f(refl_sim)
    between_sim = f(between_sim)

    # 确保邻居矩阵A的大小与当前批处理大小一致
    assert A.size(0) == z1.size(0), "邻接矩阵的大小应与样本数量一致"
    assert torch.all(torch.diag(A) == 0), "邻居矩阵的对角线应全为0"
    # 确保邻居矩阵A的对角线为0（邻居不包含自身）
    A = A.to(z1.device)

    # 计算分子：邻居的跨模态相似性指数和
    numerator_neighbors = torch.sum(between_sim * A, dim=1)

    # 提取样本自身的跨模态相似性
    numerator_self = between_sim.diag()

    # 合并分子部分
    numerator = numerator_neighbors + numerator_self

    # 计算分母：同一模态的不同样本之间的相似性指数和 + 跨模态的所有相似性指数和
    # 去除对角线（自我相似性）
    refl_sim_no_diag = refl_sim - torch.diag(torch.diag(refl_sim))#去除同一模态相似性矩阵的对角线（排除自身相似性
    same_modality_sum = torch.sum(refl_sim_no_diag, dim=1)#计算同一模态的相似性总和（排除自身）
    cross_modality_sum = torch.sum(between_sim, dim=1)#计算跨模态的相似性总和（包含自身）
    denominator = same_modality_sum + cross_modality_sum#将同一模态（排除自身）和跨模态（包含自身）的相似性总和相加，形成最终分母

    # 避免除零错误
    denominator = torch.clamp(denominator, min=1e-8)

    # 计算负对数损失
    loss = -torch.log(numerator / denominator)

    return loss.mean()


def CL_loss_SCprotein(h1: torch.Tensor, h2: torch.Tensor, A: torch.Tensor, temperature=0.5):
    """结合空间邻接矩阵的对比损失函数"""
    # 计算两个方向的半损失
    l1 = semi_loss(h1, h2, temperature)
    l2 = semi_loss(h2, h1, temperature)
    # 返回平均损失
    return (l1 + l2) * 0.5
def CL_loss_with_Spatial_graph(h1: torch.Tensor, h2: torch.Tensor, A: torch.Tensor, temperature):#, temperature=0.3
    """结合空间图结构的对比损失函数"""
    # 计算两个方向的损失：h1与h2，以及h2与h1
    l1 = semi_loss_with_neighbors1(h1, h2, A, temperature)#semi_loss_with_neighbors
    l2 = semi_loss_with_neighbors1(h2, h1, A, temperature)

    # 取平均
    loss = (l1 + l2) * 0.5
    return loss.mean()


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