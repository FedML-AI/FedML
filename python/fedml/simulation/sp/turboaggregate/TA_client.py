from torch import nn

from ..fedavg.client import Client


class TA_Client(Client):
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
        self.isdrop = isdrop
