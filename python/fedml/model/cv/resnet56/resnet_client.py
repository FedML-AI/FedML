"""
ResNet for CIFAR-10/100 Dataset.

Reference:
1. https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
2. https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua
3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385

"""

import torch
import torch.nn as nn

__all__ = ["ResNet"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 convolution with padding.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int): Stride for the convolution operation.
        groups (int): Number of groups for grouped convolution.
        dilation (int): Dilation factor for the convolution operation.

    Returns:
        nn.Conv2d: A 3x3 convolutional layer.
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
        stride (int): Stride for the convolution operation.

    Returns:
        nn.Conv2d: A 1x1 convolutional layer.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic building block for ResNet.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the convolutional layers. Default is 1.
        downsample (nn.Module, optional): Downsample layer for shortcut connection. Default is None.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        base_width (int, optional): Width of each group. Default is 64.
        dilation (int, optional): Dilation factor for convolution. Default is 1.
        norm_layer (nn.Module, optional): Normalization layer. Default is None.

    Attributes:
        expansion (int): Expansion factor for the block.

    Example:
        block = BasicBlock(64, 128, stride=2)
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
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
        Forward pass through the BasicBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
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
    Bottleneck building block for ResNet.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the convolutional layers. Default is 1.
        downsample (nn.Module, optional): Downsample layer for shortcut connection. Default is None.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        base_width (int, optional): Width of each group. Default is 64.
        dilation (int, optional): Dilation factor for convolution. Default is 1.
        norm_layer (nn.Module, optional): Normalization layer. Default is None.

    Attributes:
        expansion (int): Expansion factor for the block.

    Example:
        block = Bottleneck(256, 512, stride=2)
    """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
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
        Forward pass through the Bottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
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
        ResNet model architecture.

        Args:
            block (nn.Module): The block type to use for constructing layers (e.g., BasicBlock or Bottleneck).
            layers (list of int): List specifying the number of blocks in each layer.
            num_classes (int, optional): Number of output classes. Default is 10.
            zero_init_residual (bool, optional): Whether to initialize the last BN in each residual branch to zero. Default is False.
            groups (int, optional): Number of groups for grouped convolution. Default is 1.
            width_per_group (int, optional): Width of each group. Default is 64.
            replace_stride_with_dilation (list of bool, optional): List indicating if stride should be replaced with dilation. Default is None.
            norm_layer (nn.Module, optional): Normalization layer. Default is None.
            KD (bool, optional): Knowledge distillation flag. Default is False.

        Attributes:
            expansion (int): Expansion factor for the blocks.

        Example:
            # Example architecture for a ResNet-18 model with 2 blocks in each layer.
            model = ResNet(BasicBlock, [2, 2, 2, 2])
            # Alternatively, for a ResNet-50 model with 3 blocks in each layer.
            model = ResNet(Bottleneck, [3, 4, 6, 3])
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

        # initialization is defined here:https://github.com/pytorch/pytorch/tree/master/torch/nn/modules
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )  # init: kaiming_uniform
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        # self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.fc = nn.Linear(16 * block.expansion, num_classes)
        # self.fc = nn.Linear(32 * block.expansion, num_classes)

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
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
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
        Forward pass through the ResNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits and extracted features.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        # x = self.maxpool(x)
        extracted_features = x

        x = self.layer1(x)  # B x 16 x 32 x 32
        # x = self.layer2(x)  # B x 32 x 16 x 16
        # x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B x 64 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 64
        logits = self.fc(x_f)  # B x num_classes
        return logits, extracted_features


def resnet5_56(c, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-5-56 model.

    Args:
        c (int): Number of output classes.
        pretrained (bool): If True, returns a model pre-trained.
        path (str, optional): Path to a pre-trained checkpoint. Default is None.
        **kwargs: Additional keyword arguments to pass to the ResNet constructor.

    Returns:
        nn.Module: A ResNet-5-56 model.

    Example:
        # Create a ResNet-5-56 model with 10 output classes.
        model = resnet5_56(10)
    """
    
    model = ResNet(BasicBlock, [1, 2, 2], num_classes=c, **kwargs)
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


def resnet8_56(c, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-8-56 model.

    Args:
        c (int): Number of output classes.
        pretrained (bool): If True, returns a model pre-trained.
        path (str, optional): Path to a pre-trained checkpoint. Default is None.
        **kwargs: Additional keyword arguments to pass to the ResNet constructor.

    Returns:
        nn.Module: A ResNet-8-56 model.

    Example:
        # Create a ResNet-8-56 model with 10 output classes.
        model = resnet8_56(10)
    """

    model = ResNet(Bottleneck, [2, 2, 2], num_classes=c, **kwargs)
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
