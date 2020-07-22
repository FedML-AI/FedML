import torch.nn as nn


class VFLClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(VFLClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim, bias=bias),
        )

    def forward(self, x):
        return self.classifier(x)
