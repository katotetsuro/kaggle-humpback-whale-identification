import torch.nn as nn


class DebugModel(nn.Module):
    def __init__(self, **args):
        print('デバッグ用のめっちゃ単純なモデルを使います')
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3, 5)

    def forward(self, x):
        h = self.gap(x)
        h = self.fc(h[:, :, 0, 0])
        return h
