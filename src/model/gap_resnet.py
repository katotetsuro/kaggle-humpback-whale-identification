import torchvision
import torch.nn as nn


class GapResnet(nn.Module):
    def __init__(self, n_class=5005):
        super().__init__()
        resnet = torchvision.models.resnet18(True)
        self.extractor = nn.Sequential(*list(resnet.children())[:8])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512, n_class)

    def forward(self, x):
        h = self.extractor(x)
        h = self.gap(h)
        h = self.head(h[:, :, 0, 0])

        return h
