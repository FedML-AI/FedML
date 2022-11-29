import torch


class LogisticRegression_Cifar10(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression_Cifar10, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Flatten images into vectors
        # print(f"size = {x.size()}")
        x = x.view(x.size(0), -1)
        outputs = torch.sigmoid(self.linear(x))
        # except:
        #     print(x.size())
        #     import pdb
        #     pdb.set_trace()
        return outputs
