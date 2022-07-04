import MNN
nn = MNN.nn
F = MNN.expr


class Lenet_cifar(nn.Module):
    """construct a lenet 5 model"""
    def __init__(self):
        super(Lenet_cifar, self).__init__()
        self.conv1 = nn.conv(3, 6, [5, 5])
        self.conv2 = nn.conv(6, 16, [5, 5])
        self.fc1 = nn.linear(400, 120)
        self.fc2 = nn.linear(120, 84)
        self.fc3 = nn.linear(84, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool(x, [2, 2], [2, 2])
        x = F.relu(self.conv2(x))
        x = F.max_pool(x, [2, 2], [2, 2])
        # MNN use NC4HW4 format for convs, so we need to convert it to NCHW before entering other ops
        x = F.convert(x, F.NCHW)
        x = F.reshape(x, [0, -1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, 1)
        return x
    
class Lenet_mnist(nn.Module):
    """construct a lenet 5 model"""
    def __init__(self):
        super(Lenet_mnist, self).__init__()
        self.conv1 = nn.conv(1, 20, [5, 5])
        self.conv2 = nn.conv(20, 50, [5, 5])
        self.fc1 = nn.linear(800, 500)
        self.fc2 = nn.linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool(x, [2, 2], [2, 2])
        x = F.relu(self.conv2(x))
        x = F.max_pool(x, [2, 2], [2, 2])
        # MNN use NC4HW4 format for convs, so we need to convert it to NCHW before entering other ops
        x = F.convert(x, F.NCHW)
        x = F.reshape(x, [0, -1])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, 1)
        return x

net = Lenet_mnist()
# net.train(True)
input_var = MNN.expr.placeholder([1, 1, 28, 28], MNN.expr.NCHW)
predicts = net.forward(input_var)
# print(predicts)
F.save([predicts], "lenet_mnist.mnn")


