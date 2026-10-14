"""ResNet18 迁移学习：ImageNet 预训练权重，首层改为单通道灰度输入。"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Expression(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

    def forward(self, x):
        return self.net(x)
