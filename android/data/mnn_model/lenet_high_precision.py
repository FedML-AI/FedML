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

# save to onnx format
net = LeNet5()

# # Optional: load init parameters from mnn model
# parm_dic = read_mnn_as_tensor_dict("lenet_mnist_params.mnn")
# for k, v in net.state_dict().items():
#     if k == "conv1.weight":
#         net.state_dict()[k].copy_(parm_dic[6])
#     elif k == "conv1.bias":
#         net.state_dict()[k].copy_(parm_dic[7])
#     elif k == "conv2.weight":
#         net.state_dict()[k].copy_(parm_dic[4])
#     elif k == "conv2.bias":
#         net.state_dict()[k].copy_(parm_dic[5])
#     elif k == "fc1.weight":
#         net.state_dict()[k].copy_(parm_dic[3])
#     elif k == "fc1.bias":
#         # turn tensor shape from [1, 500] to [500]
#         parm_dic[2] = parm_dic[2].reshape(500)
#         net.state_dict()[k].copy_(parm_dic[2])
#     elif k == "fc2.weight":
#         net.state_dict()[k].copy_(parm_dic[1])
#     elif k == "fc2.bias":
#         # turn tensor shape from [1, 10] to [10]
#         parm_dic[0] = parm_dic[0].reshape(10)
#         net.state_dict()[k].copy_(parm_dic[0])

torch.onnx.export(net, torch.randn(1, 1, 28, 28), "lenet.onnx", verbose=True)
# Then Use mnnconvert -f ONNX --modelFile $1 --MNNModel $2

# tensor = np.loadtxt("mnist_train.csv", dtype=np.float32, delimiter=",")
# for itr_num in range(1):
#     # get first 784 columns as the image
#     feature = tensor[itr_num, 0:784]
#     feature = feature / 255.0
#     example = torch.from_numpy(feature).reshape(1, 1, 28, 28)
#     # set example to float32
#     example = example.type(torch.float32)
#     np.set_printoptions(precision=6)
#     # print(f"example: {example.numpy()}")

#     predict = net(example)
#     print(f"predict: {predict}")

#     # get label from the last column
#     label = tensor[itr_num, 784]
#     # newTarget = torch.zeros(1, 10, dtype=torch.float32)
#     # newTarget[0, int(label)] = 1.0

#     newTarget = torch.tensor([int(label)], dtype=torch.long)
#     # compute loss using cross entropy
#     loss = torch.nn.functional.cross_entropy(predict, newTarget)
#     print(f"before forward, loss: {loss}")
#     # backpropagation
#     optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     predict = net(example)  # compute loss again
#     print(f"after back, example: {predict}")
#     loss = torch.nn.functional.cross_entropy(predict, newTarget)
#     print(f"after backward, loss: {loss}")
#     exit()
