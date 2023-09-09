import MNN

nn = MNN.nn
F = MNN.expr


class ResBlock(nn.Module):
    """
    Residual Block for a ResNet-like architecture.

    This class defines a basic residual block with two convolutional layers and batch normalization.
    
    Args:
        in_planes (int): Number of input channels.
        planes (int): Number of output channels (number of filters in the convolutional layers).
        stride (int): Stride value for the first convolutional layer (default is 1).

    Returns:
        torch.Tensor: Output tensor from the residual block.
    """

    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        """
        Forward pass of the Residual Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the residual block.
        """

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x  # Skip connection
        out = F.relu(out)
        return out


class ResBlock_conv_shortcut(nn.Module):
    """
    Residual Block with Convolutional Shortcuts for a ResNet-like architecture.

    This class defines a residual block with convolutional shortcuts. It consists of two convolutional layers
    with batch normalization and a convolutional shortcut connection.

    Args:
        in_planes (int): Number of input channels.
        planes (int): Number of output channels (number of filters in the convolutional layers).
        stride (int): Stride value for the first convolutional layer (default is 1).

    Returns:
        torch.Tensor: Output tensor from the residual block.
    """

    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock_conv_shortcut, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv_shortcut = nn.Conv2d(
            in_planes, planes, kernel_size=1, stride=stride, bias=False
        )
        self.bn_shortcut = nn.BatchNorm2d(planes)

    def forward(self, x):
        """
        Forward pass of the Residual Block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the residual block.
        """

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.bn_shortcut(self.conv_shortcut(x))
        out += shortcut  # Skip connection with convolutional shortcut
        out = F.relu(out)
        return out


class Resnet20(nn.Module):
    """
    ResNet-20 implementation for image classification.

    This class defines a ResNet-20 architecture with convolutional blocks and shortcuts.
    It consists of four stages, each containing convolutional blocks.
    
    Args:
        num_classes (int): Number of output classes.

    Returns:
        torch.Tensor: Output tensor representing class probabilities.
    """

    def __init__(self, num_classes=10):
        super(Resnet20, self).__init__()

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = ResBlock(16, 16, 1)
        self.layer2 = ResBlock(16, 16, 1)
        self.layer3 = ResBlock(16, 16, 1)

        self.layer4 = ResBlock_conv_shortcut(16, 32, 2)
        self.layer5 = ResBlock(32, 32, 1)
        self.layer6 = ResBlock(32, 32, 1)

        self.layer7 = ResBlock_conv_shortcut(32, 64, 2)
        self.layer8 = ResBlock(64, 64, 1)
        self.layer9 = ResBlock(64, 64, 1)

        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Forward pass of the ResNet-20 model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing class probabilities.
        """

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)

        x = F.avg_pool2d(x, kernel_size=8, stride=8)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        out = F.softmax(x, dim=1)
        return out


def create_mnn_resnet20_model(mnn_file_path):
    net = Resnet20()
    input_var = MNN.expr.placeholder([1, 3, 32, 32], MNN.expr.NCHW)
    predicts = net.forward(input_var)
    F.save([predicts], mnn_file_path)
