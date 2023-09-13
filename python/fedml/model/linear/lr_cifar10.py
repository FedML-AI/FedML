import torch


class LogisticRegression_Cifar10(torch.nn.Module):
    """
    Logistic Regression Model for CIFAR-10 Image Classification.

    This class implements a logistic regression model for classifying images in the CIFAR-10 dataset.

    Args:
        input_dim (int): The input dimension, typically representing the number of features in each input sample
                         (flattened image vectors).
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

    Example:
        To create a CIFAR-10 logistic regression model with 3072 input features (32x32x3 images):
        >>> model = LogisticRegression_Cifar10(input_dim=3072, output_dim=10)

    Forward Method:
        The forward method computes the forward pass of the Logistic Regression model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            outputs (Tensor): Output tensor of shape (batch_size, output_dim) with class probabilities.

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

        # except:
        #     print(x.size())
        #     import pdb
        #     pdb.set_trace()

        outputs = torch.sigmoid(self.linear(x))
        return outputs
