"""
ResNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

"""
import logging

import torch
import torch.nn as nn

__all__ = ["ResNet"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 convolution with padding.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride for the convolution. Default is 1.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        dilation (int, optional): Dilation rate for the convolution. Default is 1.

    Returns:
        nn.Conv2d: A 3x3 convolutional layer.

    Example:
        # Create a 3x3 convolution with 64 input channels, 128 output channels, and a stride of 2.
        conv_layer = conv3x3(64, 128, stride=2)
    """

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """
    1x1 convolution.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride for the convolution. Default is 1.

    Returns:
        nn.Conv2d: A 1x1 convolutional layer.

    Example:
        # Create a 1x1 convolution with 64 input channels and 128 output channels.
        conv_layer = conv1x1(64, 128)
    """

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic building block for a ResNet.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the convolution. Default is 1.
        downsample (nn.Module, optional): Downsample layer for shortcut connection. Default is None.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        base_width (int, optional): Base width for width calculation. Default is 64.
        dilation (int, optional): Dilation rate for the convolution. Default is 1.
        norm_layer (nn.Module, optional): Normalization layer. Default is None.

    Attributes:
        expansion (int): Expansion factor for the number of output channels.

    Example:
        # Create a BasicBlock with 64 input channels and 128 output channels.
        block = BasicBlock(64, 128)
    """

    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass of the BasicBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Example:
            # Forward pass through a BasicBlock.
            output = block(input_tensor)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck building block for a ResNet.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the convolution. Default is 1.
        downsample (nn.Module, optional): Downsample layer for shortcut connection. Default is None.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        base_width (int, optional): Base width for width calculation. Default is 64.
        dilation (int, optional): Dilation rate for the convolution. Default is 1.
        norm_layer (nn.Module, optional): Normalization layer. Default is None.

    Attributes:
        expansion (int): Expansion factor for the number of output channels.

    Example:
        # Create a Bottleneck with 64 input channels and 128 output channels.
        bottleneck = Bottleneck(64, 128)
    """

    expansion = 4

    def __init__(
        self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass of the Bottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Example:
            # Forward pass through a Bottleneck block.
            output = bottleneck(input_tensor)
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        KD=False,
    ):
        """
        ResNet model implementation.

        Args:
            block (nn.Module): The type of block to use in the network (e.g., BasicBlock or Bottleneck).
            layers (list of int): Number of blocks in each layer of the network.
            num_classes (int): Number of output classes. Default is 10.
            zero_init_residual (bool): Whether to zero-init the last BN in each residual branch. Default is False.
            groups (int): Number of groups for grouped convolution. Default is 1.
            width_per_group (int): Number of channels per group for grouped convolution. Default is 64.
            replace_stride_with_dilation (list of bool): List indicating whether to replace 2x2 stride with dilation.
            norm_layer (nn.Module): Normalization layer. Default is None.
            KD (bool): Whether to enable knowledge distillation. Default is False.

        Attributes:
            block (nn.Module): The type of block used in the network.
            layers (list of int): Number of blocks in each layer of the network.
            num_classes (int): Number of output classes.
            zero_init_residual (bool): Whether to zero-init the last BN in each residual branch.
            groups (int): Number of groups for grouped convolution.
            base_width (int): Base width for width calculation.
            dilation (int): Dilation rate for the convolution.
            conv1 (nn.Conv2d): The initial convolutional layer.
            bn1 (nn.BatchNorm2d): Batch normalization layer after the initial convolution.
            relu (nn.ReLU): ReLU activation function.
            layer1 (nn.Sequential): The first layer of the network.
            layer2 (nn.Sequential): The second layer of the network.
            layer3 (nn.Sequential): The third layer of the network.
            avgpool (nn.AdaptiveAvgPool2d): Adaptive average pooling layer.
            fc (nn.Linear): Fully connected layer for classification.
            KD (bool): Whether knowledge distillation is enabled.

        Example:
            # Create a ResNet-18 model with 10 output classes.
            resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        """
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.KD = KD
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        Helper function to create a layer of blocks.

        Args:
            block (nn.Module): The type of block to use.
            planes (int): Number of output channels for the layer.
            blocks (int): Number of blocks in the layer.
            stride (int, optional): Stride for the convolution. Default is 1.
            dilate (bool, optional): Whether to use dilation. Default is False.

        Returns:
            nn.Sequential: A sequential container of blocks.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Example:
            # Forward pass through a ResNet model.
            output = resnet(input_tensor)
        """
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)  # B x 16 x 32 x 32
        # x = self.maxpool(x)
        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        x = self.fc(x_f)  # B x num_classes
        return x


def resnet56_server(c, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    This function creates a ResNet-110 model for server-side applications with the specified number of output classes.

    Args:
        c (int): Number of output classes.
        pretrained (bool): If True, returns a model pre-trained.
        path (str, optional): Path to a pre-trained model checkpoint. Default is None.
        **kwargs: Additional keyword arguments to pass to the ResNet model constructor.

    Returns:
        nn.Module: A ResNet-110 model.

    Example:
        # Create a ResNet-110 model with 10 output classes.
        model = resnet56_server(10)
    """
    logging.info("path = " + str(path))
    model = ResNet(Bottleneck, [6, 6, 6], num_classes=c, **kwargs)
    if pretrained:
        checkpoint = torch.load(path)
        state_dict = checkpoint["state_dict"]

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model
