import torch
from torch import nn

from ....core.alg_frame.client_trainer import ClientTrainer


class MyModelTrainer(ClientTrainer):
    def __init__(self, model, args=None, enable_cuda_rpc=False):
        super().__init__(model, args)
        self.enable_cuda_rpc = enable_cuda_rpc

    def get_model_params(self):
        if self.enable_cuda_rpc:
            return self.model.state_dict()
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        return False
