import MNN

nn = MNN.nn
F = MNN.expr


class Lenet5(nn.Module):
    """
    LeNet-5 convolutional neural network model.

    This class defines the LeNet-5 architecture for image classification.

    Args:
        None

    Returns:
        torch.Tensor: Model predictions.
    """

    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv1 = nn.conv(1, 20, [5, 5])
        self.conv2 = nn.conv(20, 50, [5, 5])
        self.fc1 = nn.linear(800, 500)
        self.fc2 = nn.linear(500, 10)

    def forward(self, x):
        """
        Forward pass of the LeNet-5 model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Model predictions.
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool(x, [2, 2], [2, 2])
        x = F.relu(self.conv2(x))
        x = F.max_pool(x, [2, 2], [2, 2])
        # MNN use NC4HW4 format for convs, so we need to convert it to NCHW before entering other ops
        x = F.convert(x, F.NCHW)
        x = F.reshape(x, [0, -1])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, 1)
        return x


def create_mnn_lenet5_model(mnn_file_path):
    """
    Create and save a LeNet-5 model in the MNN format.

    Args:
        mnn_file_path (str): The path to save the MNN model file.

    Note:
        This function assumes you have a LeNet-5 model class defined in a 'lenet5' module.
        The LeNet-5 model class should have a 'forward' method that takes an input tensor and returns predictions.

    Example:
        To create and save a LeNet-5 model to 'lenet5.mnn':
        >>> create_mnn_lenet5_model('lenet5.mnn')

    """
    # Create an instance of the LeNet-5 model
    net = Lenet5()

    # Define an input tensor with the desired shape (1 batch, 1 channel, 28x28)
    input_var = MNN.expr.placeholder([1, 1, 28, 28], MNN.expr.NCHW)

    # Perform a forward pass to generate predictions
    predicts = net.forward(input_var)

    # Save the model to the specified file path
    F.save([predicts], mnn_file_path)
    