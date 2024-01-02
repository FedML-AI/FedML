import sys
import os
import torch.onnx
import torch.nn as nn
import numpy as np
F_TORCH = torch.nn.functional
# from read_params_from_mnn import read_mnn_as_tensor_dict
# torch.set_printoptions(precision=7, sci_mode=False)
def self_imp_softmax(x, dim):
    """Self-implemented softmax function to imporve the precision of softmax."""
    x = x - x.max(dim=dim, keepdim=True)[0]
    x = torch.exp(x)
    x = x / x.sum(dim=dim, keepdim=True, dtype=torch.float32)
    return x

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, dtype=torch.float32)
        self.conv2 = nn.Conv2d(20, 50, 5, dtype=torch.float32)
        self.fc1 = nn.Linear(800, 500, dtype=torch.float32)
        self.fc2 = nn.Linear(500, 10, dtype=torch.float32)

    def forward(self, x):
        x = F_TORCH.relu(self.conv1(x))
        x = F_TORCH.max_pool2d(x, 2, 2)
        x = F_TORCH.relu(self.conv2(x))
        x = F_TORCH.max_pool2d(x, 2, 2)

        x = x.view(x.shape[0], -1)
        x = F_TORCH.relu(self.fc1(x))
        x = self.fc2(x)
        x = self_imp_softmax(x, 1)
        return x

def torch_mnn_transform(mnn_export_dir, mnn_name):
    net = LeNet5()
    if not os.path.exists(mnn_export_dir):
        os.makedirs(mnn_export_dir)
    # save state_dict
    torch_init_params_dir = mnn_export_dir + mnn_name + ".pth"
    onnx_file_dir = mnn_export_dir + mnn_name + ".onnx"
    mnn_file_dir = mnn_export_dir + mnn_name + ".mnn"
    print("torch_init_params_dir: ", torch_init_params_dir)
    print("onnx_file_dir: ", onnx_file_dir)
    print("mnn_file_dir: ", mnn_file_dir)

    torch.save(net.state_dict(), torch_init_params_dir)
    torch.onnx.export(net, torch.randn(1, 1, 28, 28), onnx_file_dir, verbose=True)
    # excecute "mnnconvert" command to convert onnx to mnn in the terminal
    import subprocess
    command = f'mnnconvert -f ONNX --modelFile {onnx_file_dir} --MNNModel {mnn_file_dir}'
    subprocess.run(command, shell=True, text=True)

if __name__ == "__main__":
    mnn_export_dir = sys.argv[1]
    mnn_name = sys.argv[2]
    torch_mnn_transform(mnn_export_dir, mnn_name)