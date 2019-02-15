import torchvision
import torch.nn as nn
import torch


class GapResnet(nn.Module):
    def __init__(self, n_class=5005, backbone='resnet18', global_average_pooling=True):
        super().__init__()
        model_class = getattr(torchvision.models, backbone)
        resnet = model_class(True)
        self.extractor = nn.Sequential(*list(resnet.children())[:8])
        self.global_average_pooling = global_average_pooling
        s = self._get_output_dim(self.extractor)
        if global_average_pooling:
            print('Global Average Poolingで空間方向のデータを1つにまとめます')
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            input_dim = s[0]
        else:
            print('ふつうのResNet')
            input_dim = s[0] * s[1] * s[2]

        self.head = nn.Linear(input_dim, n_class)
        self.dropout = nn.Dropout()

    def forward(self, x):
        h = self.extractor(x)
        if self.global_average_pooling:
            h = self.gap(h)
        h = h.view(h.size(0), -1)
        h = self.head(h)
        h = self.dropout(h)

        return h

    def _get_output_dim(self, model):
        x = torch.zeros(1, 3, 224, 224)
        _, c, h, w = model(x).shape
        return c, h, w
