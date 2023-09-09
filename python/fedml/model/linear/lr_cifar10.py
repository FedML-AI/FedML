import torch


class LogisticRegression_Cifar10(torch.nn.Module):
    """
    Logistic Regression Model for CIFAR-10 Image Classification.

    Args:
        input_dim (int): The input dimension, typically the number of features in each input sample.
        output_dim (int): The output dimension, representing the number of classes in CIFAR-10.

    Input:
        - Input tensor of shape (batch_size, input_dim), where batch_size is the number of input samples.

    Output:
        - Output tensor of shape (batch_size, output_dim), representing class probabilities for CIFAR-10 classes.

    Architecture:
        - Linear Layer:
            - Input: input_dim neurons (flattened image vectors)
            - Output: output_dim neurons (class probabilities)
            - Activation: Sigmoid (to produce class probabilities)

    """
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression_Cifar10, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the Logistic Regression model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            outputs (Tensor): Output tensor of shape (batch_size, output_dim) with class probabilities.

        """
        # Flatten images into vectors
        # print(f"size = {x.size()}")
        x = x.view(x.size(0), -1)
        outputs = torch.sigmoid(self.linear(x))
        # except:
        #     print(x.size())
        #     import pdb
        #     pdb.set_trace()
        return outputs
