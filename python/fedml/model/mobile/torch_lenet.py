import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.fc2 = nn.Linear(500, 10)
        self.fc1 = nn.Linear(800, 500)
        self.dp = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(20, 50, (5, 5))
        self.conv1 = nn.Conv2d(1, 20, (5, 5))
        self.maxp = nn.MaxPool2d([2, 2], stride=(2, 2))
        self.rl = nn.ReLU()
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp(x)
        x = self.conv2(x)
        x = self.maxp(x)
        # add a reshape
        x = torch.reshape(x, (1, -1))
        x = self.fc1(x)
        x = self.rl(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = self.sm(x)
        return x
