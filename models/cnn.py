"""自建小型 CNN：输入 1x48x48 灰度图，输出 7 类表情 logits。"""

import torch.nn as nn


class ExpressionCNN(nn.Module):
    """3 个卷积块（Conv-BN-ReLU x2 + MaxPool + Dropout2d）+ 2 层全连接。"""

    def __init__(self, num_classes: int = 7, dropout: float = 0.5):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, kernel_size=3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.25),
            )

        # 48 -> 24 -> 12 -> 6
        self.features = nn.Sequential(block(1, 64), block(64, 128), block(128, 256))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
