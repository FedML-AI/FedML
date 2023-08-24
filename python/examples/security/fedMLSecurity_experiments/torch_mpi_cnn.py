import logging
import fedml
from fedml import FedMLRunner
from fedml.model.cv.resnet import resnet20
import wandb



if __name__ == "__main__":
    args = fedml.init()

    if args.enable_wandb:
        args.wandb_obj = wandb.init(
            entity="fedml", project="fedmlSecurity", name="lr_security", config=args
        )

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model)
    fedml_runner.run()
