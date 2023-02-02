
import MNN

import fedml
from fedml.cross_device import ServerMNN
from my_dataset import MnistDataset

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    train_dataset = MnistDataset(args.data_cache_dir, True)
    test_dataset = MnistDataset(args.data_cache_dir, False)
    train_dataloader = MNN.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = MNN.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    class_num = 10

    # load model
    model = fedml.model.create(args, output_dim=class_num)

    # start training
    server = ServerMNN(args, device, test_dataloader, None)  # for MNN, the model is loaded using MNN file
    server.run()
