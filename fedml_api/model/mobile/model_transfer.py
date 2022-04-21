import MNN
import torch
import os

from fedml_api.model.mobile.torch_lenet import LeNet
from fedml_api.model.mobile.mnn_lenet import Net
import numpy as np
import time

F = MNN.expr


def init_mnn_model(mnn_file_path):
    model = Net()
    model.train(False)
    F.save(model.parameters, mnn_file_path, False)


def mnn_pytorch(mnn_path, torch_path=None):
    T1 = time.time()
    # initialize PyTorch model
    torch_model = LeNet()
    torch_model.eval()
    pt_layers = [x.data for x in torch_model.parameters()]
    # initialize MNN model
    mnn_model = F.load_as_list(mnn_path)
    if len(mnn_model) != len(pt_layers):
        raise Exception("model format is not aligned")

    # convert from MNN to PyTorch format
    n_layer = len(mnn_model)
    for idx_layer in range(n_layer):
        mnn_model[idx_layer].fix_as_const()
        mnn_layer_weights_np_arr = mnn_model[idx_layer].read()
        idx_pt_layer = n_layer - idx_layer - 1
        if mnn_layer_weights_np_arr.shape != pt_layers[idx_pt_layer].data.shape:
            mnn_layer_weights_np_arr = mnn_layer_weights_np_arr.reshape(pt_layers[idx_pt_layer].data.shape)
        pt_layers[idx_pt_layer].data = torch.from_numpy(mnn_layer_weights_np_arr)
    i = 0
    for x in torch_model.parameters():
        x.data = pt_layers[i].data
        i = i + 1

    if torch_path is not None:
        torch.save(torch_model.state_dict(), torch_path)
    T2 = time.time()
    print("MNN to Pytorch time is ", (T2-T1)*1000, "ms")
    return torch_model


def pytorch_mnn(pt_model, mnn_ori_path, mnn_save_path):
    # initialize PyTorch model
    T1 = time.time()
    torch_model = LeNet()
    model = torch.load(pt_model)
    torch_model.load_state_dict(model, strict=False)
    torch_model.eval()
    pt_layers = [x.data for x in torch_model.parameters()]
    n_layer = len(pt_layers)
    # initialize/detect MNN model
    if os.path.exists(mnn_save_path):
        mnn_model = F.load_as_list(mnn_save_path)
        if len(mnn_model) != len(pt_layers):
            raise Exception("model format is not aligned")
    else:
        mnn_model = F.load_as_list(mnn_ori_path)
        if len(mnn_model) != len(pt_layers):
            raise Exception("model format is not aligned")

    # convert from PyTorch to MNN format
    for idx_layer in range(n_layer):
        pt_layers[idx_layer].detach()
        pt_layer_weights_np_arr = pt_layers[idx_layer].numpy()
        idx_mnn_layer = n_layer - idx_layer - 1
        # for bias weight in mnn fc layer
        if pt_layer_weights_np_arr.shape != mnn_model[idx_mnn_layer].shape:
            pt_layer_weights_np_arr = pt_layer_weights_np_arr.reshape(mnn_model[idx_mnn_layer].shape)
        mnn_model[idx_mnn_layer] = F.const(pt_layer_weights_np_arr, list(pt_layer_weights_np_arr.shape))
    # save mnn model
    F.save(mnn_model, mnn_save_path, False)
    T2 = time.time()
    print("Pytorch to MNN time is ", (T2-T1)*1000, "ms")


def mnn_vs_mnn(origin_path, save_path):
    # load origin and restored model
    origin_weight = F.load_as_list(origin_path)
    restored_weight = F.load_as_list(save_path)
    origin_model = Net()
    restored_model = Net()
    origin_model.load_parameters(origin_weight)
    restored_model.load_parameters(restored_weight)
    origin_model.train(False)
    restored_model.train(False)

    # compare output
    random_input = np.random.random((1, 1, 28, 28))
    input_var = MNN.expr.placeholder([1, 1, 28, 28])
    input_var.write(random_input.flatten().tolist())

    origin = origin_model.forward(input_var)
    restored = restored_model.forward(input_var)
    print(origin)
    print(restored)


def mnn_vs_pt(mnn_path, pt_model):
    # init models
    origin_weight = F.load_as_list(mnn_path)
    T1 = time.time()
    mnn_model = Net()
    mnn_model.load_parameters(origin_weight)
    mnn_model.train(False)
    T2 = time.time()
    print("Pytorch loading time is ",(T2-T1)*1000, "ms")
    torch_model = LeNet()
    torch_model.load_state_dict(torch.load(pt_model))
    torch_model.eval()

    # compare results
    random_input = np.random.random((1, 1, 28, 28))
    input_var = MNN.expr.placeholder([1, 1, 28, 28])
    input_var.write(random_input.flatten().tolist())
    input_pt = torch.from_numpy(random_input).float()

    origin = mnn_model.forward(input_var)
    pt = torch_model.forward(input_pt).detach().numpy()
    print(origin)
    print(pt)


if __name__ == "__main__":
    cpp_path = "../model_test/mnist.snapshot.mnn"
    py_path = "../model_test/temp.mnist.snapshot"
    save_path = "../model_test/restored.snapshot.mnn"
    test_path = "../model_test/tested.snapshot.mnn"
    torch_path = "../model_test/model.ckpt"
    init_mnn_model(test_path)
    pt_model = mnn_pytorch(test_path, torch_path)
    pytorch_mnn(torch_path, test_path, save_path)
    print("origin mnn vs saved mnn")
    mnn_vs_mnn(test_path, save_path)
    print("origin mnn vs pytorch")
    mnn_vs_pt(test_path, torch_path)
    print("saved mnn vs pytorch")
    mnn_vs_pt(test_path, torch_path)


