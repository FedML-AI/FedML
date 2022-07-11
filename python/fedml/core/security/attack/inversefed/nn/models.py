"""Define basic models and translate some torchvision stuff."""
"""Stuff from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py."""
import torch
import torchvision
import torch.nn as nn

from torchvision.models.resnet import Bottleneck
from .revnet import iRevNet
from .densenet import _DenseNet, _Bottleneck

from collections import OrderedDict
import numpy as np
from ..utils import set_random_seed




def construct_model(model, num_classes=10, seed=None, num_channels=3, modelkey=None):
    """Return various models."""
    if modelkey is None:
        if seed is None:
            model_init_seed = np.random.randint(0, 2**32 - 10)
        else:
            model_init_seed = seed
    else:
        model_init_seed = modelkey
    set_random_seed(model_init_seed)

    if model in ['ConvNet', 'ConvNet64']:
        model = ConvNet(width=64, num_channels=num_channels, num_classes=num_classes)
    elif model == 'ConvNet8':
        model = ConvNet(width=64, num_channels=num_channels, num_classes=num_classes)
    elif model == 'ConvNet16':
        model = ConvNet(width=64, num_channels=num_channels, num_classes=num_classes)
    elif model == 'ConvNet32':
        model = ConvNet(width=64, num_channels=num_channels, num_classes=num_classes)
    elif model == 'BeyondInferringMNIST':
        model = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(1, 32, 3, stride=2, padding=1)),
            ('relu0', torch.nn.LeakyReLU()),
            ('conv2', torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            ('relu1', torch.nn.LeakyReLU()),
            ('conv3', torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            ('relu2', torch.nn.LeakyReLU()),
            ('conv4', torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)),
            ('relu3', torch.nn.LeakyReLU()),
            ('flatt', torch.nn.Flatten()),
            ('linear0', torch.nn.Linear(12544, 12544)),
            ('relu4', torch.nn.LeakyReLU()),
            ('linear1', torch.nn.Linear(12544, 10)),
            ('softmax', torch.nn.Softmax(dim=1))
        ]))
    elif model == 'BeyondInferringCifar':
        model = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)),
            ('relu0', torch.nn.LeakyReLU()),
            ('conv2', torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            ('relu1', torch.nn.LeakyReLU()),
            ('conv3', torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            ('relu2', torch.nn.LeakyReLU()),
            ('conv4', torch.nn.Conv2d(128, 256, 3, stride=1, padding=1)),
            ('relu3', torch.nn.LeakyReLU()),
            ('flatt', torch.nn.Flatten()),
            ('linear0', torch.nn.Linear(12544, 12544)),
            ('relu4', torch.nn.LeakyReLU()),
            ('linear1', torch.nn.Linear(12544, 10)),
            ('softmax', torch.nn.Softmax(dim=1))
        ]))
    elif model == 'MLP':
        width = 1024
        model = torch.nn.Sequential(OrderedDict([
            ('flatten', torch.nn.Flatten()),
            ('linear0', torch.nn.Linear(3072, width)),
            ('relu0', torch.nn.ReLU()),
            ('linear1', torch.nn.Linear(width, width)),
            ('relu1', torch.nn.ReLU()),
            ('linear2', torch.nn.Linear(width, width)),
            ('relu2', torch.nn.ReLU()),
            ('linear3', torch.nn.Linear(width, num_classes))]))
    elif model == 'TwoLP':
        width = 2048
        model = torch.nn.Sequential(OrderedDict([
            ('flatten', torch.nn.Flatten()),
            ('linear0', torch.nn.Linear(3072, width)),
            ('relu0', torch.nn.ReLU()),
            ('linear3', torch.nn.Linear(width, num_classes))]))
    elif model == 'ResNet20':
        model = ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16)
    elif model == 'ResNet20-nostride':
        model = ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16,
                       strides=[1, 1, 1, 1])
    elif model == 'ResNet20-10':
        model = ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16 * 10)
    elif model == 'ResNet20-4':
        model = ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16 * 4)
    elif model == 'ResNet20-4-unpooled':
        model = ResNet(torchvision.models.resnet.BasicBlock, [3, 3, 3], num_classes=num_classes, base_width=16 * 4,
                       pool='max')
    elif model == 'ResNet28-10':
        model = ResNet(torchvision.models.resnet.BasicBlock, [4, 4, 4], num_classes=num_classes, base_width=16 * 10)
    elif model == 'ResNet32':
        model = ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16)
    elif model == 'ResNet32-10':
        model = ResNet(torchvision.models.resnet.BasicBlock, [5, 5, 5], num_classes=num_classes, base_width=16 * 10)
    elif model == 'ResNet44':
        model = ResNet(torchvision.models.resnet.BasicBlock, [7, 7, 7], num_classes=num_classes, base_width=16)
    elif model == 'ResNet56':
        model = ResNet(torchvision.models.resnet.BasicBlock, [9, 9, 9], num_classes=num_classes, base_width=16)
    elif model == 'ResNet110':
        model = ResNet(torchvision.models.resnet.BasicBlock, [18, 18, 18], num_classes=num_classes, base_width=16)
    elif model == 'ResNet18':
        model = ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=num_classes, base_width=64)
    elif model == 'ResNet34':
        model = ResNet(torchvision.models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=num_classes, base_width=64)
    elif model == 'ResNet50':
        model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes, base_width=64)
    elif model == 'ResNet50-2':
        model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=num_classes, base_width=64 * 2)
    elif model == 'ResNet101':
        model = ResNet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], num_classes=num_classes, base_width=64)
    elif model == 'ResNet152':
        model = ResNet(torchvision.models.resnet.Bottleneck, [3, 8, 36, 3], num_classes=num_classes, base_width=64)
    elif model == 'MobileNet':
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # cifar adaptation, cf.https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenetv2.py
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        model = torchvision.models.MobileNetV2(num_classes=num_classes,
                                               inverted_residual_setting=inverted_residual_setting,
                                               width_mult=1.0)
        model.features[0] = torchvision.models.mobilenet.ConvBNReLU(num_channels, 32, stride=1)  # this is fixed to width=1
    elif model == 'MNASNet':
        model = torchvision.models.MNASNet(1.0, num_classes=num_classes, dropout=0.2)
    elif model == 'DenseNet121':
        model = torchvision.models.DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                                            num_init_features=64, bn_size=4, drop_rate=0, num_classes=num_classes,
                                            memory_efficient=False)
    elif model == 'DenseNet40':
        model = _DenseNet(_Bottleneck, [6, 6, 6, 0], growth_rate=12, num_classes=num_classes)
    elif model == 'DenseNet40-4':
        model = _DenseNet(_Bottleneck, [6, 6, 6, 0], growth_rate=12 * 4, num_classes=num_classes)
    elif model == 'SRNet3':
        model = SRNet(upscale_factor=3, num_channels=num_channels)
    elif model == 'SRNet1':
        model = SRNet(upscale_factor=1, num_channels=num_channels)
    elif model == 'iRevNet':
        if num_classes <= 100:
            in_shape = [num_channels, 32, 32]  # only for cifar right now
            model = iRevNet(nBlocks=[18, 18, 18], nStrides=[1, 2, 2],
                            nChannels=[16, 64, 256], nClasses=num_classes,
                            init_ds=0, dropout_rate=0.1, affineBN=True,
                            in_shape=in_shape, mult=4)
        else:
            in_shape = [3, 224, 224]  # only for imagenet
            model = iRevNet(nBlocks=[6, 16, 72, 6], nStrides=[2, 2, 2, 2],
                            nChannels=[24, 96, 384, 1536], nClasses=num_classes,
                            init_ds=2, dropout_rate=0.1, affineBN=True,
                            in_shape=in_shape, mult=4)
    elif model == 'LeNetZhu':
        model = LeNetZhu(num_channels=num_channels, num_classes=num_classes)
    else:
        raise NotImplementedError('Model not implemented.')

    print(f'Model initialized with random key {model_init_seed}.')
    return model, model_init_seed


class ResNet(torchvision.models.ResNet):
    """ResNet generalization for CIFAR thingies."""

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, base_width=64, replace_stride_with_dilation=None,
                 norm_layer=None, strides=[1, 2, 2, 2], pool='avg'):
        """Initialize as usual. Layers and strides are scriptable."""
        super(torchvision.models.ResNet, self).__init__()  # nn.Module
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer


        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False, False]
        if len(replace_stride_with_dilation) != 4:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 4-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups

        self.inplanes = base_width
        self.base_width = 64  # Do this to circumvent BasicBlock errors. The value is not actually used.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layers = torch.nn.ModuleList()
        width = self.inplanes
        for idx, layer in enumerate(layers):
            self.layers.append(self._make_layer(block, width, layer, stride=strides[idx], dilate=replace_stride_with_dilation[idx]))
            width *= 2

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if pool == 'avg' else nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Linear(width // 2 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for layer in self.layers:
            x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class ConvNet(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', torch.nn.BatchNorm2d(1 * width)),
            ('relu0', torch.nn.ReLU()),

            ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', torch.nn.BatchNorm2d(2 * width)),
            ('relu1', torch.nn.ReLU()),

            ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', torch.nn.BatchNorm2d(2 * width)),
            ('relu2', torch.nn.ReLU()),

            ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', torch.nn.BatchNorm2d(4 * width)),
            ('relu3', torch.nn.ReLU()),

            ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', torch.nn.BatchNorm2d(4 * width)),
            ('relu4', torch.nn.ReLU()),

            ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', torch.nn.BatchNorm2d(4 * width)),
            ('relu5', torch.nn.ReLU()),

            ('pool0', torch.nn.MaxPool2d(3)),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', torch.nn.BatchNorm2d(4 * width)),
            ('relu7', torch.nn.ReLU()),

            ('pool1', torch.nn.MaxPool2d(3)),
            ('flatten', torch.nn.Flatten()),
            ('linear', torch.nn.Linear(36 * width, num_classes))
        ]))

    def forward(self, input):
        return self.model(input)


class LeNetZhu(nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""

    def __init__(self, num_classes=10, num_channels=3):
        """3-Layer sigmoid Conv with large linear layer."""
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, num_classes)
        )
        for module in self.modules():
            self.weights_init(module)

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(out)
        return out
