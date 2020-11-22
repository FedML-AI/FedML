import os, sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_api.model.cv.xception import *
from fedml_api.model.cv.batchnorm_utils import SynchronizedBatchNorm2d

# from batchnorm_utils import SynchronizedBatchNorm2d


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()

        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation
            
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()

        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, dilation=dilations[3], BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()        

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()        

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLabv3_plus(nn.Module):
    def __init__(self, backbone='xception', nInputChannels=3, n_classes=21, output_stride=16, pretrained=False, freeze_bn=False, sync_bn=False, _print=True):
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Backbone: Xception")
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(output_stride))
            print("Number of Input Channels: {}".format(nInputChannels))
        super(DeepLabv3_plus, self).__init__()

        if backbone == 'drn':
            output_stride = 8        

        if sync_bn == True:
            BatchNorm2d = SynchronizedBatchNorm2d
        else:
            BatchNorm2d = nn.BatchNorm2d

        self.backbone = self.build_backbone(backbone=backbone, n_channels=nInputChannels, output_stride=output_stride, BatchNorm=BatchNorm2d, pretrained=pretrained)
        self.aspp = self.build_aspp(backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm2d)
        self.decoder = self.build_decoder(num_classes=n_classes, backbone=backbone, BatchNorm=BatchNorm2d)

        self.freeze_bn = freeze_bn

        if freeze_bn:
            self._freeze_bn()

    def forward(self, input):

        # with open('input_sample.npy', 'wb') as f:
        #     np.save(f, input.cpu().numpy(), allow_pickle=True)
        #     f.close()
            
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, BatchNorm2d):
                m.eval()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def build_backbone(backbone='xception', n_channels=3, output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=True):
        print(backbone)
        if backbone == 'xception':
            return AlignedXception(inplanes = n_channels, output_stride = output_stride, BatchNorm=BatchNorm, pretrained=pretrained)
        else:
            raise NotImplementedError

    @staticmethod
    def build_aspp(backbone, output_stride, BatchNorm):
        return ASPP(backbone, output_stride, BatchNorm)        

    @staticmethod
    def build_decoder(num_classes, backbone, BatchNorm):
        return Decoder(num_classes, backbone, BatchNorm)        

    def get_1x_lr_params(self):
            modules = [self.backbone]
            for i in range(len(modules)):
                for m in modules[i].named_modules():
                    if self.freeze_bn:
                        if isinstance(m[1], nn.Conv2d):
                            for p in m[1].parameters():
                                if p.requires_grad:
                                    yield p
                    else:
                        if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                                or isinstance(m[1], nn.BatchNorm2d):
                            for p in m[1].parameters():
                                if p.requires_grad:
                                    yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLabv3_plus(nInputChannels=3, n_classes=3, output_stride=16, pretrained=True, _print=True)
    model.eval()
    image = torch.randn(16,3,513,513)
    # with open('input_sample.npy', 'rb') as f:
    #     image = torch.tensor(np.load(f, allow_pickle=True))
    with torch.no_grad():
        output = model.forward(image)
    print(output.size())