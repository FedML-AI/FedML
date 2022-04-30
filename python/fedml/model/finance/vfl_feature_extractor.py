import torch.nn as nn


class VFLFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VFLFeatureExtractor, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim), nn.LeakyReLU()
        )
        self.output_dim = output_dim

    def forward(self, x):
        return self.classifier(x)

    def get_output_dim(self):
        return self.output_dim
