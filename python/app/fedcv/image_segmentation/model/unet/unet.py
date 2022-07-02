"""
Refer https://github.com/qubvel/segmentation_models.pytorch
"""
import os, sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from fedml.model.cv.batchnorm_utils import SynchronizedBatchNorm2d
from unet_utils import Conv2dReLU, Activation, Attention
from ..resnet import ResNet101


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        # print(x.shape)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # print(x.shape)
        if skip is not None:
            # print(skip.shape)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class FeatureExtractor(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm, pretrained):
        super(FeatureExtractor, self).__init__()
        self.backbone = self.build_backbone(
            backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm, pretrained=pretrained
        )

    def forward(self, input):
        features = self.backbone(input)
        return features

    @staticmethod
    def build_backbone(
        backbone="resnet", output_stride=16, BatchNorm=nn.BatchNorm2d, pretrained=False, model_name="unet"
    ):

        if backbone == "resnet":
            return ResNet101(output_stride, BatchNorm, model_name, pretrained=False)
        else:
            raise NotImplementedError


class UNet(nn.Module):
    def __init__(
        self,
        backbone="resnet",
        encoder_depth=5,
        encoder_weights="imagenet",
        decoder_use_batchnorm=True,
        encoder_out_channels=[3, 64, 256, 512, 1024, 2048],
        decoder_channels=[256, 128, 64, 32, 16],
        decoder_attention_type=None,
        in_channels=3,
        n_classes=21,
        activation=None,
        aux_params=None,
        output_stride=16,
        pretrained=False,
        sync_bn=False,
    ):
        super(UNet, self).__init__()

        if sync_bn == True:
            BatchNorm2d = SynchronizedBatchNorm2d
        else:
            BatchNorm2d = nn.BatchNorm2d

        self.n_classes = n_classes

        logging.info(
            "Constructing UNet model with Backbone {0}, number of classes {1}, output stride {2}".format(
                backbone, n_classes, output_stride
            )
        )

        self.encoder = FeatureExtractor(
            backbone=backbone, output_stride=output_stride, BatchNorm=BatchNorm2d, pretrained=pretrained
        )

        self.decoder = UnetDecoder(
            encoder_channels=encoder_out_channels[: encoder_depth + 1],
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if backbone.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=n_classes,
            activation=activation,
            kernel_size=3,
        )

    def initialize_decoder(self, module):
        for m in module.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def initialize_head(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def initialize(self):
        self.initialize_decoder(self.decoder)
        self.initialize_head(self.segmentation_head)

    def forward(self, x):
        features = self.encoder(x)
        # logging.info("After obtaining features from backbone : {}".format(features.shape))
        decoder_output = self.decoder(*features)
        # logging.info("After executing decoder : {}".format(decoder_output.shape))
        masks = self.segmentation_head(decoder_output)
        # print("Final segmentation masks : {}".format(masks.shape))
        return masks

    def get_1x_lr_params(self):
        modules = [self.encoder.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
                else:
                    if (
                        isinstance(m[1], nn.Conv2d)
                        or isinstance(m[1], SynchronizedBatchNorm2d)
                        or isinstance(m[1], nn.BatchNorm2d)
                    ):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.decoder, self.segmentation_head]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
                else:
                    if (
                        isinstance(m[1], nn.Conv2d)
                        or isinstance(m[1], SynchronizedBatchNorm2d)
                        or isinstance(m[1], nn.BatchNorm2d)
                    ):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


if __name__ == "__main__":
    image = torch.randn(16, 3, 512, 512)
    model = UNet(backbone="resnet", output_stride=16, n_classes=1, pretrained=False)
    with torch.no_grad():
        output = model.forward(image)
    # print(output.size())
