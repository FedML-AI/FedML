from __future__ import print_function

import time

import MNN
import mnist

nn = MNN.nn
F = MNN.expr


import numpy as np

from fedml.model import create_mnn_lenet5_model

from model_utils import read_mnn_as_tensor_dict, write_tensor_dict_to_mnn
import fedml


class MnistDataset(MNN.data.Dataset):
    def __init__(self, training_dataset=True):
        super(MnistDataset, self).__init__()
        self.is_training_dataset = training_dataset
        if self.is_training_dataset:
            self.data = (
                mnist.train_images() / 255.0
            )  # downloading happens in this function
            self.labels = mnist.train_labels()
        else:
            self.data = mnist.test_images() / 255.0
            self.labels = mnist.test_labels()

    def __getitem__(self, index):
        dv = F.const(
            self.data[index].flatten().tolist(), [1, 28, 28], F.data_format.NCHW
        )
        dl = F.const([self.labels[index]], [], F.data_format.NCHW, F.dtype.uint8)
        # first for inputs, and may have many inputs, so it's a list
        # second for targets, also, there may be more than one targets
        return [dv], [dl]

    def __len__(self):
        # size of the dataset
        if self.is_training_dataset:
            return 60000
        else:
            return 10000


class MyAggregator:
    def __init__(self, test_global):
        self.test_global = test_global

    def test_on_server_for_all_clients(self, mnn_file_path):
        # load global model from MNN
        var_map = F.load_as_dict(mnn_file_path)
        input_dicts, output_dicts = F.get_inputs_and_outputs(var_map)
        input_names = [n for n in input_dicts.keys()]
        output_names = [n for n in output_dicts.keys()]
        input_vars = [input_dicts[n] for n in input_names]
        output_vars = [output_dicts[n] for n in output_names]
        module = MNN.nn.load_module(input_vars, output_vars, False)

        module.train(False)
        self.test_global.reset()

        correct = 0
        for i in range(self.test_global.iter_number):
            example = self.test_global.next()
            input_data = example[0]
            output_target = example[1]
            data = input_data[0]  # which input, model may have more than one inputs
            label = output_target[0]  # also, model may have more than one outputs

            result = module.forward(data)
            predict = F.argmax(result, 1)
            predict = np.array(predict.read())

            label_test = np.array(label.read())
            correct += np.sum(label_test == predict)

            target = F.one_hot(F.cast(label, F.int), 10, 1, 0)
            loss = nn.loss.cross_entropy(result, target)

        print("test acc: ", correct * 100.0 / self.test_global.size, "%")
        print("test loss: ", loss.read())


def aggregate(model_dict, sample_num_dict, worker_num):
    start_time = time.time()
    model_list = []
    training_num = 0

    for idx in range(worker_num):
        fedml.logger.info("self.model_dict[idx] = {}".format(model_dict[idx]))
        mnn_file_path = model_dict[idx]
        tensor_params_dict = read_mnn_as_tensor_dict(mnn_file_path)
        model_list.append((sample_num_dict[idx], tensor_params_dict))
        training_num += sample_num_dict[idx]
    fedml.logger.info("training_num = {}".format(training_num))
    fedml.logger.info("len of self.model_dict[idx] = " + str(len(model_dict)))

    # logger.info("################aggregate: %d" % len(model_list))
    (num0, averaged_params) = model_list[0]
    for k in averaged_params.keys():
        for i in range(0, len(model_list)):
            local_sample_number, local_model_params = model_list[i]
            w = local_sample_number / training_num
            fedml.logger.info("w = {}".format(w))
            if i == 0:
                averaged_params[k] = local_model_params[k] * w
            else:
                averaged_params[k] += local_model_params[k] * w
    end_time = time.time()
    fedml.logger.info("aggregate time cost: %d" % (end_time - start_time))
    print(len(averaged_params.keys()))
    return averaged_params


if __name__ == "__main__":
    train_dataset = MnistDataset(True)
    test_dataset = MnistDataset(False)
    train_dataloader = MNN.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = MNN.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    agg = MyAggregator(test_dataloader)
    print("dataloader is done")

    global_model_file_path = "./global_model.mnn"
    create_mnn_lenet5_model(global_model_file_path)

    model1_path = "fedml_189_0_126-095b441b3c044623a36724dde12c2ac5"
    model_dict = dict()
    model_dict[0] = model1_path
    agg.test_on_server_for_all_clients(model1_path)
    model1_dict = read_mnn_as_tensor_dict(model1_path)
    global_model_dict = read_mnn_as_tensor_dict(global_model_file_path)
    print(len(model1_dict))
    print(len(global_model_dict))

    c = F.load_as_list(model1_path)
    print(len(c))
    d = F.load_as_list(global_model_file_path)
    print(len(d))

    fedml.logger.info("--------------------------")

    model2_path = "fedml_189_0_126-119cb458d2f84523a28a3d514afbd5af"
    model_dict[1] = model2_path
    e = F.load_as_list(model2_path)
    print(len(e))
    agg.test_on_server_for_all_clients(model2_path)

    model2_path = read_mnn_as_tensor_dict(model2_path)
    print(len(model2_path))

    fedml.logger.info("--------------------------")

    sample_num_dict = dict()
    sample_num_dict[0] = 600
    sample_num_dict[1] = 600

    # load model

    worker_num = 2
    global_model_params = aggregate(model_dict, sample_num_dict, worker_num)

    write_tensor_dict_to_mnn(global_model_file_path, global_model_params)

    agg.test_on_server_for_all_clients(global_model_file_path)
