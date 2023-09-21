import torch.nn as nn

class LeNet(nn.Module):
    """
    LeNet-5 is a convolutional neural network architecture that was designed for handwritten and machine-printed character
    recognition tasks. This implementation includes four convolutional layers and one fully connected layer.

    Args:
        None

    Attributes:
        body (nn.Sequential): The convolutional layers of the LeNet model.
        fc (nn.Sequential): The fully connected layer of the LeNet model.

    """
    def __init__(self):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(nn.Linear(768, 10))

    def forward(self, x):
        """
        Forward pass of the LeNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).

        """
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
