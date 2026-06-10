"""Exact AlexNet architecture (Krizhevsky et al., NeurIPS 2012).

Adapted for Imagenette: 10 output classes, AdaptiveAvgPool2d(6) to handle
128×128 input (paper uses 224×224 → 6×6 after Conv5+MaxPool).

Configurable: width_multiplier (1.0 = paper), dropout (0.5 = paper).
No LRN — omitted per design decision (obsolete since BatchNorm).
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, dropout=0.5):
        super().__init__()
        w = lambda c: max(1, int(c * width_mult))

        self.conv1 = nn.Conv2d(3, w(96), kernel_size=11, stride=4, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(w(96), w(256), kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(w(256), w(384), kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(w(384), w(384), kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(w(384), w(256), kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(6)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.fc6 = nn.Linear(w(256) * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Paper: conv layers N(0, 0.01), FC layers N(0, 0.005)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.005)

        # Paper: bias = 1 for Conv2, Conv4, Conv5 and all FC layers.
        # Bias = 0 (default) for Conv1, Conv3.
        for name, m in [
            ("conv2", self.conv2), ("conv4", self.conv4), ("conv5", self.conv5),
            ("fc6", self.fc6), ("fc7", self.fc7), ("fc8", self.fc8),
        ]:
            nn.init.constant_(m.bias, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3(x))

        x = self.relu(self.conv4(x))

        x = self.relu(self.conv5(x))
        x = self.pool5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(self.relu(self.fc6(x)))
        x = self.dropout(self.relu(self.fc7(x)))
        x = self.fc8(x)
        return x


def build_alexnet(config):
    """Factory: build AlexNet from experiment config dict.

    config keys:
        width_mult (float): 1.0 = paper, 0.5 = reduced width
        dropout (float):  0.5 = paper, 0.0 = no dropout
        num_classes (int): default 10
    """
    return AlexNet(
        num_classes=config.get("num_classes", 10),
        width_mult=config.get("width_mult", 1.0),
        dropout=config.get("dropout", 0.5),
    )
