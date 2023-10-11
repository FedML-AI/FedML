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


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    Create a 3x3 convolution layer with padding.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride for the convolution. Default is 1.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        dilation (int, optional): Dilation rate for the convolution. Default is 1.

    Returns:
        nn.Conv2d: A 3x3 convolution layer.

    Example:
        # Create a 3x3 convolution layer with 64 input channels and 128 output channels.
        conv_layer = conv3x3(64, 128)
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
    Create a 1x1 convolution layer.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride for the convolution. Default is 1.

    Returns:
        nn.Conv2d: A 1x1 convolution layer.

    Example:
        # Create a 1x1 convolution layer with 64 input channels and 128 output channels.
        conv_layer = conv1x1(64, 128)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic building block for ResNet.

    Args:
        inplanes (int): Number of input channels.
        planes (int): Number of output channels.
        stride (int, optional): Stride for the convolution. Default is 1.
        downsample (nn.Module, optional): Downsample layer. Default is None.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        base_width (int, optional): Base width for grouped convolution. Default is 64.
        dilation (int, optional): Dilation rate for the convolution. Default is 1.
        norm_layer (nn.Module, optional): Normalization layer. Default is None.

    Attributes:
        expansion (int): The expansion factor of the block.

    Example:
        # Create a BasicBlock with 64 input channels and 128 output channels.
        block = BasicBlock(64, 128)
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
        stride (int, optional): Stride for the convolution. Default is 1.
        downsample (nn.Module, optional): Downsample layer. Default is None.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        base_width (int, optional): Base width for grouped convolution. Default is 64.
        dilation (int, optional): Dilation rate for the convolution. Default is 1.
        norm_layer (nn.Module, optional): Normalization layer. Default is None.

    Attributes:
        expansion (int): The expansion factor of the block (default is 4).

    Example:
        # Create a Bottleneck block with 64 input channels and 128 output channels.
        block = Bottleneck(64, 128)
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
    """
    ResNet model architecture for image classification.

    Args:
        block (nn.Module): The building block for the network (e.g., BasicBlock or Bottleneck).
        layers (list): List of integers specifying the number of blocks in each layer.
        num_classes (int, optional): Number of classes for classification. Default is 10.
        zero_init_residual (bool, optional): If True, zero-initialize the last BN in each residual branch.
            Default is False.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        width_per_group (int, optional): Base width for grouped convolution. Default is 64.
        replace_stride_with_dilation (list or None, optional): List of booleans specifying if the 2x2 stride
            should be replaced with dilated convolution in each layer. Default is None.
        norm_layer (nn.Module, optional): Normalization layer. Default is None.
        KD (bool, optional): Knowledge distillation flag. Default is False.

    Attributes:
        expansion (int): The expansion factor of the building block (default is 4).

    Example:
        # Create a ResNet-56 model with 10 output classes.
        model = ResNet(Bottleneck, [6, 6, 6], num_classes=10)
    """
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
        # self.maxpool = nn.MaxPool2d()
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        # self.fc = nn.Linear(16 * block.expansion, num_classes)
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
        """
        Create a layer of blocks for the ResNet model.

        Args:
            block (nn.Module): The building block for the layer (e.g., BasicBlock or Bottleneck).
            planes (int): The number of output channels for the layer.
            blocks (int): The number of blocks to stack in the layer.
            stride (int, optional): The stride for the layer's convolutional operations. Default is 1.
            dilate (bool, optional): If True, apply dilated convolutions in the layer. Default is False.

        Returns:
            nn.Sequential: A sequential container of blocks representing the layer.

        Example:
            # Create a layer of 2 Bottleneck blocks with 64 output channels and stride 1.
            layer = self._make_layer(Bottleneck, 64, 2, stride=1)
        """
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
        Forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W), where B is the batch size, C is the number of
                channels, H is the height, and W is the width.

        Returns:
            torch.Tensor: The output tensor of shape (B, num_classes) representing class logits.
            torch.Tensor: Extracted features before the classification layer, of shape (B, C, H, W).
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        extracted_features = x

        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)
        x = self.layer3(x)

        # output here
        x = self.avgpool(x)  # B x 16 x 1 x 1
        x_f = x.view(x.size(0), -1)  # B x 16
        logits = self.fc(x_f)  # B x num_classes
        return logits, extracted_features


def resnet32_pretrained(c, pretrained=False, path=None, **kwargs):
    """
    Constructs a pre-trained ResNet-32 model.

    Args:
        c (int): The number of output classes.
        pretrained (bool): If True, returns a model pre-trained on a given path.
        path (str, optional): The path to the pre-trained model checkpoint. Default is None.
        **kwargs: Additional keyword arguments to pass to the ResNet model.

    Returns:
        nn.Module: A pre-trained ResNet-32 model.

    Example:
        # Create a pre-trained ResNet-32 model with 10 output classes.
        model = resnet32_pretrained(10, pretrained=True, path='pretrained_resnet32.pth')
    """

    model = ResNet(BasicBlock, [5, 5, 5], num_classes=c, **kwargs)
    if pretrained:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        state_dict = checkpoint["state_dict"]

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:]  # remove 'module.' of dataparallel
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model


def resnet56_pretrained(c, pretrained=False, path=None, **kwargs):
    """
    Constructs a pre-trained ResNet-56 model.

    Args:
        c (int): The number of output classes.
        pretrained (bool): If True, returns a model pre-trained on a given path.
        path (str, optional): The path to the pre-trained model checkpoint. Default is None.
        **kwargs: Additional keyword arguments to pass to the ResNet model.

    Returns:
        nn.Module: A pre-trained ResNet-56 model.

    Example:
        # Create a pre-trained ResNet-56 model with 10 output classes.
        model = resnet56_pretrained(10, pretrained=True, path='pretrained_resnet56.pth')
    """
    logging.info("path = " + str(path))
    model = ResNet(Bottleneck, [6, 6, 6], num_classes=c, **kwargs)
    if pretrained:
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        state_dict = checkpoint["state_dict"]

        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of dataparallel
            # name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    return model
