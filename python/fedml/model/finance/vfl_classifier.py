import torch.nn as nn


class VFLClassifier(nn.Module):
    """
    Virtual Federated Learning (VFL) Classifier Model.

    Args:
        input_dim (int): The input dimension, typically the number of features in each input sample.
        output_dim (int): The output dimension, representing the number of classes.

    Input:
        - Input tensor of shape (batch_size, input_dim), where batch_size is the number of input samples.

    Output:
        - Output tensor of shape (batch_size, output_dim), representing class predictions or scores.

    Architecture:
        - Linear Layer:
            - Input: input_dim neurons
            - Output: output_dim neurons (typically the number of classes)

    """
    def __init__(self, input_dim, output_dim, bias=True):
        super(VFLClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim, bias=bias),
        )

    def forward(self, x):
        """
        Forward pass of the VFL Classifier model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            predictions (Tensor): Output tensor of shape (batch_size, output_dim) with class predictions or scores.

        """
        return self.classifier(x)
