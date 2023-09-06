from torch import nn

from ..fedavg.client import Client


class TA_Client(Client):
    """
    A subclass of the Client class for a specific type of client.

    Args:
        client_idx (int): The index of the client.
        local_training_data: The local training data for the client.
        local_test_data: The local test data for the client.
        local_sample_number: The number of local samples.
        args: Additional arguments.
        device: The computing device (e.g., 'cuda' or 'cpu').
        model_trainer: The model trainer for this client.
    """
    def __init__(
            self,
            client_idx,
            local_training_data,
            local_test_data,
            local_sample_number,
            args,
            device,
            model_trainer
    ):
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device

        self.criterion = nn.CrossEntropyLoss().to(device)

        self.isdrop = False
        self.model_trainer = model_trainer

        # self.buffer_in = np.zeros(dtype='int')
        # self.buffer_out = np.zeros(dtype='int')

    def set_dropout(self, isdrop):
        """
        Set the dropout flag for this client.

        Args:
            isdrop (bool): Whether to enable dropout for this client.
        """
        self.isdrop = isdrop
