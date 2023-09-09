import torch
import torch.nn as nn


class LeNet(nn.Module):
    """
    LeNet-5 Convolutional Neural Network model for image classification.

    Args:
        None

    Input:
        - Input tensor of shape (batch_size, 1, 32, 32), where batch_size is the number of input samples.

    Output:
        - Output tensor of shape (batch_size, 10), representing class probabilities for 10 classes.

    Architecture:
        - Convolutional Layer 1:
            - Input: 1 channel (grayscale image)
            - Output: 20 feature maps
            - Kernel size: 5x5
            - Activation: ReLU
            - Max Pooling: 2x2
        - Convolutional Layer 2:
            - Input: 20 feature maps
            - Output: 50 feature maps
            - Kernel size: 5x5
            - Activation: ReLU
            - Max Pooling: 2x2
        - Fully Connected Layer 1:
            - Input: 800 neurons (flattened 50x4x4 from previous layer)
            - Output: 500 neurons
            - Activation: ReLU
            - Dropout: 50% dropout rate
        - Fully Connected Layer 2:
            - Input: 500 neurons
            - Output: 10 neurons (class probabilities)
            - Activation: Softmax

    """

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
