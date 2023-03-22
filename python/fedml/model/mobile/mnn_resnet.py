import MNN

nn = MNN.nn
F = MNN.expr


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.conv(
            in_planes,
            planes,
            kernel_size=[3, 3],
            stride=[stride, stride],
            padding=[1, 1],
            bias=False,
            padding_mode=MNN.expr.Padding_Mode.SAME,
        )
        self.bn1 = nn.batch_norm(planes)
        self.conv2 = nn.conv(
            planes,
            planes,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            bias=False,
            padding_mode=MNN.expr.Padding_Mode.SAME,
        )
        self.bn2 = nn.batch_norm(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x
        out = F.relu(out)
        return out


class ResBlock_conv_shortcut(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResBlock_conv_shortcut, self).__init__()
        self.conv1 = nn.conv(
            in_planes,
            planes,
            kernel_size=[3, 3],
            stride=[stride, stride],
            padding=[1, 1],
            bias=False,
            padding_mode=MNN.expr.Padding_Mode.SAME,
        )
        self.bn1 = nn.batch_norm(planes)
        self.conv2 = nn.conv(
            planes,
            planes,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            bias=False,
            padding_mode=MNN.expr.Padding_Mode.SAME,
        )
        self.bn2 = nn.batch_norm(planes)

        self.conv_shortcut = nn.conv(
            in_planes,
            planes,
            kernel_size=[1, 1],
            stride=[stride, stride],
            bias=False,
            padding_mode=MNN.expr.Padding_Mode.SAME,
        )
        self.bn_shortcut = nn.batch_norm(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.bn_shortcut(self.conv_shortcut(x))
        out = F.relu(out)
        return out


class Resnet20(nn.Module):
    def __init__(self, num_classes=10):
        super(Resnet20, self).__init__()

        self.conv1 = nn.conv(
            3,
            16,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            bias=False,
            padding_mode=MNN.expr.Padding_Mode.SAME,
        )
        self.bn1 = nn.batch_norm(16)

        self.layer1 = ResBlock(16, 16, 1)
        self.layer2 = ResBlock(16, 16, 1)
        self.layer3 = ResBlock(16, 16, 1)

        self.layer4 = ResBlock_conv_shortcut(16, 32, 2)
        self.layer5 = ResBlock(32, 32, 1)
        self.layer6 = ResBlock(32, 32, 1)

        self.layer7 = ResBlock_conv_shortcut(32, 64, 2)
        self.layer8 = ResBlock(64, 64, 1)
        self.layer9 = ResBlock(64, 64, 1)

        self.fc = nn.linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1.forward(x)
        x = self.layer2.forward(x)
        x = self.layer3.forward(x)
        # print(x.shape)
        x = self.layer4.forward(x)
        x = self.layer5.forward(x)
        x = self.layer6.forward(x)
        # print(x.shape)
        x = self.layer7.forward(x)
        x = self.layer8.forward(x)
        x = self.layer9.forward(x)
        # print(x.shape)
        x = F.avg_pool(x, kernel=[8, 8], stride=[8, 8])
        x = F.convert(x, F.NCHW)
        x = F.reshape(x, [0, -1])
        x = self.fc(x)
        out = F.softmax(x, 1)
        return out


def create_mnn_resnet20_model(mnn_file_path):
    net = Resnet20()
    input_var = MNN.expr.placeholder([1, 3, 32, 32], MNN.expr.NCHW)
    predicts = net.forward(input_var)
    F.save([predicts], mnn_file_path)
