import torch
class LogisticRegression(torch.nn.Module):
    """
    Logistic Regression model for binary classification.

    This class defines a logistic regression model with a single linear layer followed by a sigmoid activation function
    for binary classification tasks.

    Args:
        input_dim (int): The dimensionality of the input features.
        output_dim (int): The number of output classes, which should be 1 for binary classification.

    Example:
        # Create a logistic regression model for binary classification
        input_dim = 10
        output_dim = 1
        model = LogisticRegression(input_dim, output_dim)

    Forward Method:
        The forward method computes the output of the model for a given input.

    Example:
        # Forward pass with input tensor 'x'
        input_tensor = torch.tensor([0.1, 0.2, 0.3, ..., 0.9])
        output = model(input_tensor)

    Note:
        - For binary classification, the `output_dim` should be set to 1.
        - The `forward` method applies a sigmoid activation to the linear output, producing values in the range [0, 1].

    """

    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Forward pass of the logistic regression model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).

        Example:
            # Forward pass with input tensor 'x'
            input_tensor = torch.tensor([0.1, 0.2, 0.3, ..., 0.9])
            output = model(input_tensor)
        """
        outputs = torch.sigmoid(self.linear(x))
        return outputs

