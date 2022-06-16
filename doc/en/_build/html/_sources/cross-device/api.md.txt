# FedML BeeHive API Reference

FedML BeeHive (the cross-device FL) currently support `ServerMNN` related APIs, which operates a FL server compatible with our FedML Android SDK/APP.
```python

import MNN
import fedml
import wandb
from fedml.cross_device import ServerMNN
from fedml.model import create_mnn_lenet5_model

from my_dataset import MnistDataset

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    if args.enable_wandb:
        args.wandb_obj = wandb.init(
            entity="fedml", project="mobile", name="MNN-Mobile", config=args
        )

    # init device
    device = fedml.device.get_device(args)

    # load data
    train_dataset = MnistDataset(True)
    test_dataset = MnistDataset(False)
    train_dataloader = MNN.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = MNN.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    class_num = 10

    # load model
    create_mnn_lenet5_model(args.global_model_file_path)

    # start training
    server = ServerMNN(
        args, device, test_dataloader, None
    )  # for MNN, the model is loaded using MNN file
    server.run()
```


For newly developed features, we will try to keep the form of these APIs and only add new arguments. 

To check the details of the latest definition of each API, the best resource is always the source code itself. Please check comments of each API at:
[https://github.com/FedML-AI/FedML/blob/master/python/fedml/__init__.py](https://github.com/FedML-AI/FedML/blob/master/python/fedml/__init__.py)
