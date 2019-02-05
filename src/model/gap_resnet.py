import torchvision
import torch.nn as nn
import torch


class GapResnet(nn.Module):
    def __init__(self, n_class=5005, backbone='resnet18'):
        super().__init__()
        model_class = getattr(torchvision.models, backbone)
        resnet = model_class(True)
        self.extractor = nn.Sequential(*list(resnet.children())[:8])
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(self._get_output_dim(self.extractor), n_class)
        self.dropout = nn.Dropout()

    def forward(self, x):
        h = self.extractor(x)
        h = self.gap(h)
        h = self.head(h[:, :, 0, 0])
        h = self.dropout(h)

        return h

    def _get_output_dim(self, model):
        x = torch.zeros(1, 3, 224, 224)
        _, c, _, _ = model(x).shape
        return c
