import torch
import torch.nn as nn
import torch.optim as optim


class DenseModel(nn.Module):
    """
    Dense Model with Linear Classifier.

    Args:
        input_dim (int): The input dimension, typically the number of features in each input sample.
        output_dim (int): The output dimension, representing the number of classes or features.
        learning_rate (float, optional): The learning rate for the optimizer. Default is 0.01.
        bias (bool, optional): Whether to include bias terms in the linear layer. Default is True.

    Input:
        - Input tensor of shape (batch_size, input_dim), where batch_size is the number of input samples.

    Output:
        - Output tensor of shape (batch_size, output_dim) representing the model's predictions.

    Methods:
        - forward(x): Forward pass of the model to make predictions.
        - backward(x, grads): Backward pass to compute gradients and update model parameters.

    """

    def __init__(self, input_dim, output_dim, learning_rate=0.01, bias=True):
        super(DenseModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim, bias=bias),
        )
        self.is_debug = False
        self.optimizer = optim.SGD(
            self.parameters(), momentum=0.9, weight_decay=0.01, lr=learning_rate
        )

    def forward(self, x):
        """
        Forward pass of the Dense Model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            predictions (Tensor): Output tensor of shape (batch_size, output_dim) with model predictions.

        """
        if self.is_debug:
            print("[DEBUG] DenseModel.forward")

        x = torch.tensor(x).float()
        return self.classifier(x)

    def backward(self, x, grads):
        """
        Backward pass of the Dense Model.

        Args:
            x (array-like): Input data of shape (batch_size, input_dim).
            grads (array-like): Gradients of the loss with respect to the model's output.

        Returns:
            x_grad (array-like): Gradients of the loss with respect to the input data.

        """
        if self.is_debug:
            print("[DEBUG] DenseModel.backward")

        x = torch.tensor(x, requires_grad=True).float()
        grads = torch.tensor(grads).float()
        output = self.classifier(x)
        loss = torch.sum(output * grads)  # Compute dot product for backward pass
        loss.backward()
        x_grad = x.grad.numpy()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return x_grad


class LocalModel(nn.Module):
    """
    Local Model with a Linear Classifier.

    Args:
        input_dim (int): The input dimension, typically the number of features in each input sample.
        output_dim (int): The output dimension, representing the number of classes or features.
        learning_rate (float): The learning rate for the optimizer.

    Attributes:
        output_dim (int): The output dimension of the model.

    Methods:
        forward(x): Forward pass of the model to make predictions.
        predict(x): Make predictions using the model.
        backward(x, grads): Backward pass to compute gradients and update model parameters.
        get_output_dim(): Get the output dimension of the model.

    """

    def __init__(self, input_dim, output_dim, learning_rate):
        super(LocalModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=output_dim), nn.LeakyReLU()
        )
        self.output_dim = output_dim
        self.is_debug = False
        self.learning_rate = learning_rate
        self.optimizer = optim.SGD(
            self.parameters(), momentum=0.9, weight_decay=0.01, lr=learning_rate
        )

    def forward(self, x):
        """
        Forward pass of the Local Model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            predictions (array-like): Output predictions as a numpy array.

        """
        if self.is_debug:
            print("[DEBUG] LocalModel.forward")

        x = torch.tensor(x).float()
        return self.classifier(x).detach().numpy()

    def predict(self, x):
        """
        Make predictions using the Local Model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            predictions (array-like): Output predictions as a numpy array.

        """
        if self.is_debug:
            print("[DEBUG] LocalModel.predict")

        x = torch.tensor(x).float()
        return self.classifier(x).detach().numpy()

    def backward(self, x, grads):
        """
        Backward pass of the Local Model.

        Args:
            x (array-like): Input data of shape (batch_size, input_dim).
            grads (array-like): Gradients of the loss with respect to the model's output.

        """
        if self.is_debug:
            print("[DEBUG] LocalModel.backward")

        x = torch.tensor(x).float()
        grads = torch.tensor(grads).float()
        output = self.classifier(x)
        loss = torch.sum(output * grads)  # Compute dot product for backward pass
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_output_dim(self):
        """
        Get the output dimension of the Local Model.

        Returns:
            output_dim (int): The output dimension of the model.

        """
        return self.output_dim
