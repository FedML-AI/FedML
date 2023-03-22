
import re
from turtle import forward
import MNN
nn = MNN.nn
F = MNN.expr

def make_divisible(v, divisor, min_value = None):
    if min_value is None:
        min_value = divisor
    
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, depthwise=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.conv(in_planes, planes, kernel_size=[kernel_size,kernel_size], stride=[stride,stride], bias=False, padding_mode=MNN.expr.Padding_Mode.SAME, depthwise=depthwise)
        self.bn = nn.batch_norm(planes)

    def forward(self, x):
        out = F.relu6(self.bn(self.conv(x)))
        return out

class BottleNeck(nn.Module):
    def __init__(self, in_planes, planes, stride, expand_ratio):
        super(BottleNeck, self).__init__()
        expand_planes = in_planes * expand_ratio
        
        self.use_shortcut = False
        if stride == 1 and in_planes == planes:
            self.use_shortcut = True
            # print(in_planes)
        
        self.layers = []
        if expand_ratio != 1:
            self.layers.append(ConvBnRelu(in_planes, expand_planes, 1))
            
        self.layers.append(ConvBnRelu(expand_planes, expand_planes, 3, stride, True))   
        
        self.conv = nn.conv(expand_planes, planes, kernel_size=[1,1], stride=[1,1], bias=False, padding_mode=MNN.expr.Padding_Mode.SAME)
        self.bn = nn.batch_norm(planes)
        
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)

        out = self.bn(self.conv(out))

        if self.use_shortcut:
            out += x
            
        return out
    
class MobilenetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, divisor=8):
        super(MobilenetV2, self).__init__()
        in_planes = 32
        last_planes = 1280
        
        inverted_residual_setting = [
                    [1, 16, 1, 1],
                    [6, 24, 2, 1],
                    [6, 32, 3, 2],
                    [6, 64, 4, 2],
                    [6, 96, 3, 1],
                    [6, 160, 3, 2],
                    [6, 320, 1, 1]]
        
        in_planes = make_divisible(in_planes * width_mult, divisor)
        last_planes = make_divisible(last_planes * max(1.0, width_mult), divisor)

        self.first_conv = ConvBnRelu(3, in_planes, 3, 1)
        
        self.bottle_neck_blocks = []
        for t, c, n, s in inverted_residual_setting:
            out_planes = make_divisible(c * width_mult, divisor)
            
            for i in range(n):
                stride = s if i == 0 else 1

                self.bottle_neck_blocks.append(BottleNeck(in_planes, out_planes, stride, t))
                in_planes = out_planes
                
        self.last_conv = ConvBnRelu(in_planes, last_planes, 1)
        self.dropout = nn.dropout(0.1)
        self.fc = nn.linear(last_planes, num_classes)
        
    def forward(self, x):
        x = self.first_conv.forward(x)

        for layer in self.bottle_neck_blocks:
            x = layer.forward(x)
            # print(x.shape)
            
        x = self.last_conv.forward(x)
        print(x.shape)
        x = F.avg_pool(x, kernel=[4,4], stride=[1,1])
        x = F.convert(x, F.NCHW)
        x = F.reshape(x, [0, -1])
        # x = self.dropout(x)
        x = self.fc(x)
        out = F.softmax(x, 1)
        return out
    
net = MobilenetV2()
net.train(True)
input_var = MNN.expr.placeholder([1, 3, 32, 32], MNN.expr.NC4HW4)
predicts = net.forward(input_var)
# print(predicts)
F.save([predicts], "mobilenetv2.mnn")