from .gap_resnet import GapResnet
import torch.nn as nn
import torch


class FeatureExtractor(nn.Module):
    def __init__(self, mid_dim, out_dim, backbone, normalize=True, gap=True):
        super().__init__()
        self.net = GapResnet(
            n_class=mid_dim, backbone=backbone, global_average_pooling=gap)
        self.fc = nn.Linear(mid_dim, out_dim)
        self.normalize = normalize

    def forward(self, x):
        h = self.net(x)
        h = self.fc(h)
        if self.normalize:
            l = (h**2).sum(dim=1).sqrt().reshape(-1, 1)
            h = h / l

        return h

    def freeze(self):
        for p in self.net.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.net.parameters():
            p.requires_grad = True


class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = FeatureExtractor(activation=None)

    def forward(self, anchor, positive, negative):
        a = self.net(anchor)
        p = self.net(positive)
        n = self.net(negative)
        return a, p, n
