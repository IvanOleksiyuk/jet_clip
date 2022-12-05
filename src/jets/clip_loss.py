import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data


class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=1.):
        super().__init__()
        self.logit_scale = logit_scale
        self.__loss_evol = {'train': [], 'valid': []}

    @property
    def loss_evolution(self):
        return self.__loss_evol

    def item(self):
        return self.item_

    def forward(self, embedding_1, embedding_2, valid=False):
        device = embedding_1.device
        logits_1 = self.logit_scale * embedding_1 @ embedding_2.T
        logits_2 = self.logit_scale * embedding_2 @ embedding_1.T
        num_logits = logits_1.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        loss = 0.5 * (
            F.cross_entropy(logits_1, labels) +
            F.cross_entropy(logits_2, labels)
        )
        self.__loss_evol['valid' if valid else 'train'].append(loss.item())
        self.item_ = loss.item()
        return loss


class CLIPLossNorm(nn.Module):
    def __init__(self, logit_scale=1.):
        super().__init__()
        self.logit_scale = logit_scale
        self.__loss_evol = {'train': [], 'valid': []}

    @property
    def loss_evolution(self):
        return self.__loss_evol

    def item(self):
        return self.item_

    def forward(self, embedding_1, embedding_2, valid=False):
        device = embedding_1.device
        norm = embedding_1.norm(dim=1, keepdim=True) @ \
            embedding_2.norm(dim=1, keepdim=True).T
        logits_1 = (self.logit_scale * embedding_1 @ embedding_2.T) / norm
        logits_2 = (self.logit_scale * embedding_2 @ embedding_1.T) / norm.T
        num_logits = logits_1.shape[0]
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        loss = 0.5 * (
            F.cross_entropy(logits_1, labels) +
            F.cross_entropy(logits_2, labels)
        )
        self.__loss_evol['valid' if valid else 'train'].append(loss.item())
        self.item_ = loss.item()
        return loss


class ExpLoss(nn.Module):
    def __init__(self, logit_scale=1.):
        super().__init__()

    def item(self):
        return self.item_

    def forward(self, embedding_1, embedding_2, valid=False):
        device = embedding_1.device
        loss = ((embedding_1-embedding_2)**2).mean()
        self.item_ = loss.item()
        return loss
