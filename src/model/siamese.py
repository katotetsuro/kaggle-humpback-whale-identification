from .gap_resnet import GapResnet
import torch.nn as nn
import torch


class FeatureExtractor(nn.Module):
    def __init__(self, mid_dim, out_dim, backbone):
        super().__init__()
        self.net = GapResnet(n_class=mid_dim, backbone=backbone)
        self.fc = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        h = self.net(x)
        h = self.fc(h)
        l = (h**2).sum(dim=1).sqrt().reshape(-1, 1)
        h = h / l

        return h


class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = FeatureExtractor(activation=None)

    def forward(self, anchor, positive, negative):
        a = self.net(anchor)
        p = self.net(positive)
        n = self.net(negative)
        return a, p, n
