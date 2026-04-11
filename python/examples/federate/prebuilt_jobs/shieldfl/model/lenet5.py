"""
LeNet-5：MNIST 28×28 单通道输入。
学术需求标记为 "LeNet-5 (SimpleCNN)"。
"""
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # -> 6×28×28
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)              # -> 16×10×10 (after pool)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        out = F.avg_pool2d(F.relu(self.conv1(x)), 2)   # -> (B, 6, 14, 14)
        out = F.avg_pool2d(F.relu(self.conv2(out)), 2)  # -> (B, 16, 5, 5)
        out = out.view(out.size(0), -1)                  # -> (B, 400)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return self.fc3(out)
