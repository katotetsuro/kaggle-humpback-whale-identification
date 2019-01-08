from .gap_resnet import GapResnet
import torch.nn as nn
import torch


class FeatureExtractor(nn.Module):
    def __init__(self, feature_dim=500):
        super().__init__()
        self.net = GapResnet(n_class=feature_dim)
        self.fc = nn.Linear(feature_dim, 64)

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
