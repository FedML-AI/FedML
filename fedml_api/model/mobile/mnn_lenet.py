import MNN
nn = MNN.nn
F = MNN.expr


class Net(nn.Module):
    """construct a lenet 5 model"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc2 = nn.linear(500, 10)
        self.fc1 = nn.linear(800, 500)
        self.dp = nn.dropout(0.5)
        self.conv2 = nn.conv(20, 50, [5, 5])
        self.conv1 = nn.conv(1, 20, [5, 5])

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool(x, [2, 2], [2, 2])
        x = self.conv2(x)
        x = F.max_pool(x, [2, 2], [2, 2])
        # MNN use NC4HW4 format for convs, so we need to convert it to NCHW before entering other ops
        x = F.reshape(x, [0, -1])
        x = F.convert(x, F.NCHW)
        x = F.relu(self.fc1(x))
        x = self.dp(x)
        x = self.fc2(x)
        x = F.softmax(x, 1)
        return x