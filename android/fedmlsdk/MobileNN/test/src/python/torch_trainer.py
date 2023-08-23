import sys
import os
import torch.onnx
import torch.nn as nn
import numpy as np
import copy
import MNN
F_TORCH = torch.nn.functional
F = MNN.expr
# from read_params_from_mnn import read_mnn_as_tensor_dict

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
        # x = self_imp_softmax(x, 1)
        return x

class tabular_mnist_dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.tensor = np.loadtxt(data_path, dtype=np.float32, delimiter=",")

    def __getitem__(self, index):
        feature = self.tensor[index, 0:784]
        feature = feature / 255.0
        feature = torch.from_numpy(feature).reshape(1, 1, 28, 28)
        feature = feature.type(torch.float32)
        label = self.tensor[index, 784]
        label = torch.tensor([int(label)], dtype=torch.long)
        return feature, label

    def __len__(self):
        return self.tensor.shape[0]

def loss_precision(loss, precision=5):
    return (round(loss.item(), precision))

def torch_train(train_data_path, initial_weights_path, model_name):
    net = LeNet5()
    # load initial weights
    net.load_state_dict(
        torch.load(initial_weights_path)
    )

    epoch = 1
    dataset = tabular_mnist_dataset(train_data_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for itr_num in range(epoch):
        for batch_idx, (example, target) in enumerate(data_loader):
            # turn shape from (1,1,1,28,28) to (1,1,28,28)
            print(f"Epoch: {itr_num}  {batch_idx} / {len(data_loader)}")
            example = example.squeeze(0)
            # print(f"example: {example}")
            predict = net(example)
            # print(f"predict: {predict}")

            newTarget = torch.tensor([int(target)], dtype=torch.long)
            # compute loss using cross entropy
            loss = torch.nn.functional.cross_entropy(predict, newTarget)
            print(f"Before backprop, Loss: {loss_precision(loss)}")
            # backpropagation
            optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predict = net(example)  # compute loss again
            # print(f"Aftrer foward, predict: {predict}")
            loss = torch.nn.functional.cross_entropy(predict, newTarget)

            print(f"After backprop, loss: {loss_precision(loss)}")

if __name__ == "__main__":
    train_data_dir = sys.argv[1]
    initial_weights_path = sys.argv[2]
    model_name = sys.argv[3]

    train_data_path = os.path.join(train_data_dir, "mnist_train.csv")
    torch_train(train_data_path, initial_weights_path, model_name)