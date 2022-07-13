import logging

import fedml
from fedml import FedMLRunner
from model import DeepLabV3_plus, VisionTransformer, UNet
from trainer.segmentation_trainer import SegmentationTrainer
from .data.data_loader import load


def create_model(args, model_name, output_dim):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    model_name = str(model_name).lower()
    if model_name == "unet":
        model = UNet(in_channel=3, n_classes=output_dim)
    elif model_name == "deeplabv3_plus":
        model = DeepLabV3_plus(
            backbone="mobilenet",
            nInputChannels=3,
            n_classes=output_dim,
            output_stride=16,
            pretrained=False,
            _print=True,
        )
    elif model_name in ["vit", "transunet"]:
        from .model.transunet.transunet import CONFIGS

        vit_name = "R50-ViT-B_16"
        img_size, vit_patches_size = 224, 16
        config_vit = CONFIGS[vit_name]
        config_vit.n_classes = 9
        config_vit.n_skip = 3
        if vit_name.find("R50") != -1:
            config_vit.patches.grid = (
                int(img_size / vit_patches_size),
                int(img_size / vit_patches_size),
            )
        model = VisionTransformer(
            config_vit, img_size=img_size, num_classes=config_vit.n_classes
        )
    else:
        raise Exception("such model does not exist !")

    trainer = SegmentationTrainer(model=model)
    logging.info("done")

    return model, trainer


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, class_num = load(args)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model, trainer = create_model(args, args.model, output_dim=class_num)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer)
    fedml_runner.run()
