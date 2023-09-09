import torch.nn as nn


class VFLFeatureExtractor(nn.Module):
    """
    Virtual Federated Learning (VFL) Feature Extractor Model.

    Args:
        input_dim (int): The input dimension, typically the number of features in each input sample.
        output_dim (int): The output dimension, representing the desired feature dimension.

    Input:
        - Input tensor of shape (batch_size, input_dim), where batch_size is the number of input samples.

    Output:
        - Output tensor of shape (batch_size, output_dim), representing extracted features.

    Architecture:
        - Linear Layer followed by Leaky ReLU activation:
            - Input: input_dim neurons
            - Output: output_dim neurons (representing feature dimension)

    """
    def __init__(self, input_dim, output_dim):
        super(VFLFeatureExtractor, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim), nn.LeakyReLU()
        )
        self.output_dim = output_dim

    def forward(self, x):
        """
        Forward pass of the VFL Feature Extractor model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            features (Tensor): Output tensor of shape (batch_size, output_dim) with extracted features.

        """
        return self.classifier(x)

    def get_output_dim(self):
        """
        Get the output dimension of the feature extractor.

        Returns:
            int: The output dimension (feature dimension).

        """
        return self.output_dim
