import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, 1)
        return x


model = LeNet5()
example = torch.rand(1, 1, 28, 28)

# traced_module = torch.jit.trace(model, example)
# traced_module.save("traced_lenet_model.pt")

scripted_module = torch.jit.script(model, example)
# scripted_module.save("scripted_lenet_model.pt")

# no optimization but necessary for c++ jit load
optimized_scripted_module = torch.jit.optimized_execution(scripted_module)
scripted_module._save_for_lite_interpreter("scripted_lenet_model.ptl")


# from torchsummary import summary
# summary(model, (1, 28, 28), device="cpu")

# for p in model.parameters():
#     print(p.shape)
