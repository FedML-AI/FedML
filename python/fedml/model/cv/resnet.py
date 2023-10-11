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

__all__ = ["ResNet", "resnet110"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3x3 convolution with padding.
    
    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride for the convolution. Default is 1.
        groups (int, optional): Number of groups for grouped convolution. Default is 1.
        dilation (int, optional): Dilation factor for convolution. Default is 1.
    
    Returns:
        nn.Conv2d: 3x3 convolutional layer.
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
        nn.Conv2d: 1x1 convolutional layer.
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Basic residual block used in ResNet architectures.
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
        """
        Initialize a BasicBlock instance.
        
        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int, optional): Stride for the convolution. Default is 1.
            downsample (nn.Module, optional): Downsample layer for shortcut connections. Default is None.
            groups (int, optional): Number of groups for grouped convolution. Default is 1.
            base_width (int, optional): Base width for the convolution. Default is 64.
            dilation (int, optional): Dilation factor for convolution. Default is 1.
            norm_layer (nn.Module, optional): Normalization layer. Default is None.
        """
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        # Define the convolutional layers
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        
        # Downsample layer for shortcut connections
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
        """Initialize a Bottleneck block used in ResNet architectures.
        
        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int, optional): Stride for the convolution. Default is 1.
            downsample (nn.Module, optional): Downsample layer for shortcut connections. Default is None.
            groups (int, optional): Number of groups for grouped convolution. Default is 1.
            base_width (int, optional): Base width for the convolution. Default is 64.
            dilation (int, optional): Dilation factor for convolution. Default is 1.
            norm_layer (nn.Module, optional): Normalization layer. Default is None.
        """
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        
        # Define the three convolutional layers
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        
        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)
        
        # Downsample layer for shortcut connections
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Forward pass through the Bottleneck block.
        
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
        """Initialize a ResNet model.
        
        Args:
            block (nn.Module): The residual block type, either BasicBlock or Bottleneck.
            layers (list): List of integers indicating the number of blocks in each layer.
            num_classes (int, optional): Number of output classes. Default is 10.
            zero_init_residual (bool, optional): If True, zero-initialize the last BN in each residual branch. Default is False.
            groups (int, optional): Number of groups for grouped convolution. Default is 1.
            width_per_group (int, optional): Base width for the convolution. Default is 64.
            replace_stride_with_dilation (tuple, optional): Replace stride with dilation in certain stages. Default is None.
            norm_layer (nn.Module, optional): Normalization layer. Default is None.
            KD (bool, optional): Whether to perform Knowledge Distillation. Default is False.
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
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
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
        Create a layer in the ResNet model.

        Args:
            block (nn.Module): The residual block type, either BasicBlock or Bottleneck.
            planes (int): Number of output channels for the layer.
            blocks (int): Number of residual blocks in the layer.
            stride (int, optional): The stride for the convolutional layers. Default is 1.
            dilate (bool, optional): Whether to apply dilation to the convolutional layers. Default is False.

        Returns:
            nn.Sequential: A sequential layer containing the specified number of residual blocks.
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
        """Forward pass through the ResNet model.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # B x 16 x 32 x 32
        x = self.layer1(x)  # B x 16 x 32 x 32
        x = self.layer2(x)  # B x 32 x 16 x 16
        x = self.layer3(x)  # B x 64 x 8 x 8

        x = self.avgpool(x)  # B



def resnet20(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ResNet(BasicBlock, [3, 3, 3], class_num, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model



def resnet32(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ResNet(BasicBlock, [5, 5, 5], class_num, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def resnet44(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ResNet(BasicBlock, [7, 7, 7], class_num, **kwargs)
    if pretrained:
        raise NotImplementedError
    return model


def resnet56(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    model = ResNet(Bottleneck, [6, 6, 6], class_num, **kwargs)
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


def resnet110(class_num, pretrained=False, path=None, **kwargs):
    """
    Constructs a ResNet-110 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained.
    """
    logging.info("path = " + str(path))
    model = ResNet(Bottleneck, [12, 12, 12], class_num, **kwargs)
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
