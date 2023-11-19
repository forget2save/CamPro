import torch
import torch.nn as nn


class SoftmaxHead(nn.Module):
    def __init__(self, classnum, embedding_size=512):
        super(SoftmaxHead, self).__init__()
        self.linear = nn.Linear(embedding_size, classnum)

    def forward(self, embbedings, dummy_labels=None):
        out = self.linear(embbedings)
        return out
