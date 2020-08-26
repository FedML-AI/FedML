import logging

from torch import nn

from fedml_api.standalone.fedavg.client import Client


class TA_Client(Client):
    def __init__(self, local_training_data, local_test_data, local_sample_number, args, device, client_idx):
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device

        self.criterion = nn.CrossEntropyLoss().to(device)

        self.isdrop = False

        # self.buffer_in = np.zeros(dtype='int')
        # self.buffer_out = np.zeros(dtype='int')

    def set_dropout(self, isdrop):
        self.isdrop = isdrop
