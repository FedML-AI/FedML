import torch


class LogisticRegression(torch.nn.Module):
    """
    Logistic Regression Model.

    This class implements a simple logistic regression model for binary or multi-class classification tasks.

    Args:
        input_dim (int): The input dimension, typically representing the number of features in each input sample.
        output_dim (int): The output dimension, representing the number of classes (for multi-class) or 1 (for binary).

    Input:
        - Input tensor of shape (batch_size, input_dim), where batch_size is the number of input samples.

    Output:
        - Output tensor of shape (batch_size, output_dim), representing class probabilities (for multi-class)
          or a single output (for binary).

    Architecture:
        - Linear Layer:
            - Input: input_dim neurons
            - Output: output_dim neurons
            - Activation: Sigmoid (for binary classification) or Softmax (for multi-class classification)

    Note:
        - For binary classification, set output_dim to 1.
        - For multi-class classification, output_dim should be set to the number of classes.

    Example:
        To create a binary logistic regression model with 10 input features:
        >>> model = LogisticRegression(input_dim=10, output_dim=1)

    Forward Method:
        The forward method computes the forward pass of the Logistic Regression model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            outputs (Tensor): Output tensor of shape (batch_size, output_dim) with class probabilities (for multi-class)
              or a single output (for binary).

    """
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the Logistic Regression model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            outputs (Tensor): Output tensor of shape (batch_size, output_dim) with class probabilities (for multi-class)
              or a single output (for binary).

        """

        outputs = torch.sigmoid(self.linear(x))

        return outputs
