import logging
from .deeplabV3_plus import DeepLabV3_plus
from .unet.unet import UNet
from .transunet import VisionTransformer
from fedml.simulation.mpi.fedseg.utils import count_parameters


def create_model(args):
    model_name = args.model.lower()
    if model_name == "deeplabv3_plus":
        model = DeepLabV3_plus(
            backbone=args.backbone,
            image_size=args.img_size,
            n_classes=args.class_num,
            output_stride=args.outstride,
            pretrained=args.backbone_pretrained,
            freeze_bn=args.freeze_bn,
            sync_bn=args.sync_bn,
        )

        if args.backbone_freezed:
            logging.info("Freezing Backbone")
            for param in model.feature_extractor.parameters():
                param.requires_grad = False
        elif args.backbone_pretrained:
            logging.info("Finetuning Backbone")
        else:
            logging.info("Training from Scratch")

        num_params = count_parameters(model)
        logging.info("DeepLabV3_plus Model Size : {}".format(num_params))

    elif model_name == "unet":
        model = UNet(
            backbone=args.backbone,
            output_stride=args.outstride,
            n_classes=args.class_num,
            pretrained=args.backbone_pretrained,
            sync_bn=args.sync_bn,
        )

        if args.backbone_freezed:
            logging.info("Freezing Backbone")
            for param in model.encoder.parameters():
                param.requires_grad = False
        elif args.backbone_pretrained:
            logging.info("Finetuning Backbone")
        else:
            logging.info("Training from Scratch")

        num_params = count_parameters(model)
        logging.info("Unet Model Size : {}".format(num_params))

    else:
        raise ("Not Implemented Error")

    return model
