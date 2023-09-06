from copy import deepcopy



class Client:
    def __init__(
        self, client_idx, local_training_data, local_test_data, local_sample_number, args, device, model_trainer,
    ):
        """
        Initialize a client for federated learning.

        Args:
            client_idx (int): The index of the client.
            local_training_data (torch.utils.data.DataLoader): The DataLoader for local training data.
            local_test_data (torch.utils.data.DataLoader): The DataLoader for local test data.
            local_sample_number (int): The number of local training samples.
            args: The arguments for the client.
            device (torch.device): The device to perform computations on.
            model_trainer: The model trainer used for training.
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.c_model_local = deepcopy(self.model_trainer.model).cpu()
        for name, params in self.c_model_local.named_parameters():
            params.data = params.data*0


    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        """
        Update the local dataset for the client.

        Args:
            client_idx (int): The index of the client.
            local_training_data (torch.utils.data.DataLoader): The DataLoader for local training data.
            local_test_data (torch.utils.data.DataLoader): The DataLoader for local test data.
            local_sample_number (int): The number of local training samples.
        """
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        """
        Get the number of local training samples.

        Returns:
            int: The number of local training samples.
        """
        return self.local_sample_number

    def train(self, w_global, c_model_global_param):
        """
        Perform local training for the client.

        Args:
            w_global: The global model parameters.
            c_model_global_param: The global model parameters of the central model.

        Returns:
            tuple: A tuple containing weights_delta and c_delta_para.
        """

        c_model_global_param = deepcopy(c_model_global_param)
        c_model_local_param = self.c_model_local.state_dict()
        # for name in self.c_model_global:
        #     self.c_model_global[name] = self.c_model_global[name].to(self.device)
        # self.c_model_local.to(self.device)
        # c_model_local = self.c_model_local.state_dict()
        self.model_trainer.set_model_params(deepcopy(w_global))
        iteration_cnt = self.model_trainer.train(self.local_training_data, self.device, self.args, c_model_global_param, c_model_local_param)
        weights = self.model_trainer.get_model_params()

        c_new_para = self.c_model_local.cpu().state_dict()
        # c_delta_para = deepcopy(c_new_para.state_dict())
        c_delta_para = {}
        global_model_para = w_global
        net_para = weights
        weights_delta = {}
        for key in net_para:
            c_new_para[key] = c_new_para[key] - c_model_global_param[key].cpu() + \
                (global_model_para[key] - net_para[key]) / (iteration_cnt * self.args.learning_rate)
            c_delta_para[key] = c_new_para[key] - c_model_local_param[key].cpu()
            weights_delta[key] = net_para[key] - w_global[key].cpu()

        return weights_delta, c_delta_para

    def local_test(self, b_use_test_dataset):
        """
        Perform local testing on the client's dataset.

        Args:
            b_use_test_dataset (bool): If True, use the test dataset; if False, use the training dataset.

        Returns:
            dict: A dictionary containing test metrics.
        """
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics


