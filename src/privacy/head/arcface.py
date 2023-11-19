import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class ArcFaceHead(nn.Module):
    def __init__(self, classnum, embedding_size=512, s=64.0, m=0.5):
        super(ArcFaceHead, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(classnum, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.eps = 1e-6
        
    def forward(self, embbedings, labels=None):
        emb_norm = l2_norm(embbedings)
        kernel_norm = l2_norm(self.weight)
        cosine = F.linear(emb_norm, kernel_norm)
        if labels is None:
            return cosine
        
        m_hot = torch.zeros(labels.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, labels.reshape(-1, 1), self.m)

        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)  # for stability
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi - self.eps)
        cosine_m = theta_m.cos()
        scaled_cosine_m = cosine_m * self.s
        return scaled_cosine_m

