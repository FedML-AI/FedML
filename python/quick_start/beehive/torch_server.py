import os
from os.path import expanduser

import MNN

import fedml
from fedml.cross_device import ServerMNN
from my_dataset import MnistDataset

if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # Handle argument compatibility for cross-platform
    home_dir = expanduser("~")
    args.data_cache_dir = str(args.data_cache_dir).replace('~', home_dir)
    args.data_cache_dir = str(args.data_cache_dir).replace('\\', os.sep).replace('/', os.sep)
    cur_dir = os.path.split(os.path.realpath(__file__))[0]
    args.model_file_cache_folder = str(args.model_file_cache_folder).replace('\\', os.sep).replace('/', os.sep)
    args.model_file_cache_folder = os.path.join(cur_dir, args.model_file_cache_folder)
    args.global_model_file_path = str(args.global_model_file_path).replace('\\', os.sep).replace('/', os.sep)
    args.global_model_file_path = os.path.join(cur_dir, args.global_model_file_path)
    print("data_cache_dir {}".format(args.data_cache_dir))
    print("model_file_cache_folder {}".format(args.model_file_cache_folder))
    print("global_model_file_path {}".format(args.global_model_file_path))

    # init device
    device = fedml.device.get_device(args)

    # load data
    train_dataset = MnistDataset(True, root_path=args.data_cache_dir)
    test_dataset = MnistDataset(False, root_path=args.data_cache_dir)
    train_dataloader = MNN.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = MNN.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    class_num = 10

    # load model
    model = fedml.model.create(args, output_dim=class_num)

    # start training
    server = ServerMNN(
        args, device, test_dataloader, None
    )  # for MNN, the model is loaded using MNN file
    server.run()
