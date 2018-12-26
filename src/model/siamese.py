from .gap_resnet import GapResnet
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128, activation=nn.Sigmoid()):
        super().__init__()
        self.net = GapResnet(n_class=feature_dim)
        self.activation = activation

    def forward(self, x):
        h = self.net(x)
        if self.activation:
            h = self.activation(h)
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
