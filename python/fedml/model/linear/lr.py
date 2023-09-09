import torch


class LogisticRegression(torch.nn.Module):
    """
    Logistic Regression Model.

    Args:
        input_dim (int): The input dimension, typically the number of features in each input sample.
        output_dim (int): The output dimension, representing the number of classes or a single output.

    Input:
        - Input tensor of shape (batch_size, input_dim), where batch_size is the number of input samples.

    Output:
        - Output tensor of shape (batch_size, output_dim), representing class probabilities or a single output.

    Architecture:
        - Linear Layer:
            - Input: input_dim neurons
            - Output: output_dim neurons
            - Activation: Sigmoid (for binary classification) or Softmax (for multi-class classification)
    
    Note:
        - For binary classification, output_dim is typically set to 1.
        - For multi-class classification, output_dim is the number of classes.

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
            outputs (Tensor): Output tensor of shape (batch_size, output_dim) with class probabilities or a single output.

        """
        # try:
        outputs = torch.sigmoid(self.linear(x))
        # except:
        #     print(x.size())
        #     import pdb
        #     pdb.set_trace()
        return outputs
