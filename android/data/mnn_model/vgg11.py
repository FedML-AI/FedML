import MNN
nn = MNN.nn
F = MNN.expr


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, planes):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.conv(in_planes, planes, kernel_size=[3,3], stride=[1,1], bias=False, padding=[1,1])
        self.bn = nn.batch_norm(planes)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        return out

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        self.conv1 = nn.conv(3, 64, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False, padding_mode=MNN.expr.Padding_Mode.SAME)
        self.bn1 = nn.batch_norm(64)
        self.conv2 = nn.conv(64, 128, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False, padding_mode=MNN.expr.Padding_Mode.SAME)
        self.bn2 = nn.batch_norm(128)
        self.conv3 = nn.conv(128, 256, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False, padding_mode=MNN.expr.Padding_Mode.SAME)
        self.bn3 = nn.batch_norm(256)
        self.conv4 = nn.conv(256, 256, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False, padding_mode=MNN.expr.Padding_Mode.SAME)
        self.bn4 = nn.batch_norm(256)
        self.conv5 = nn.conv(256, 512, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False, padding_mode=MNN.expr.Padding_Mode.SAME)
        self.bn5 = nn.batch_norm(512)
        self.conv6 = nn.conv(512, 512, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False, padding_mode=MNN.expr.Padding_Mode.SAME)
        self.bn6 = nn.batch_norm(512)
        self.conv7 = nn.conv(512, 512, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False, padding_mode=MNN.expr.Padding_Mode.SAME)
        self.bn7 = nn.batch_norm(512)
        self.conv8 = nn.conv(512, 512, kernel_size=[3,3], stride=[1,1], padding=[1,1], bias=False, padding_mode=MNN.expr.Padding_Mode.SAME)
        self.bn8 = nn.batch_norm(512)
        
        self.fc = nn.linear(512, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool(x, [2, 2], [2, 2])
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool(x, [2, 2], [2, 2])
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool(x, [2, 2], [2, 2])
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool(x, [2, 2], [2, 2])
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool(x, [2, 2], [2, 2])
        print(x.shape)
        # x = F.avg_pool(x, kernel=[1,1], stride=[1,1])
        # x = F.convert(x, F.NCHW)
        x = F.reshape(x, [0, -1])
        x = self.fc(x)
        x = F.softmax(x, 1)
        
        return x
        
net = VGG11()
net.train(True)
input_var = MNN.expr.placeholder([1, 3, 32, 32], MNN.expr.NC4HW4)
predicts = net.forward(input_var)
# print(predicts)
F.save([predicts], "vgg11.mnn")