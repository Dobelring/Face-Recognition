"""模型定义与构建入口。"""

import torch.nn as nn

from .cnn import ExpressionCNN
from .labels import EMOTION_CLASSES, EMOTION_CN, cn
from .resnet import ResNet18Expression

MODEL_TYPES = ("cnn", "resnet18")


def build_model(model_type: str, num_classes: int = 7, pretrained: bool = False) -> nn.Module:
    """按名称构建模型。pretrained 仅对 resnet18 生效（ImageNet 迁移学习）。"""
    if model_type == "cnn":
        return ExpressionCNN(num_classes, dropout=0.5)
    if model_type == "resnet18":
        return ResNet18Expression(num_classes, pretrained=pretrained)
    raise ValueError(f"未知模型类型: {model_type}，可选 {MODEL_TYPES}")
