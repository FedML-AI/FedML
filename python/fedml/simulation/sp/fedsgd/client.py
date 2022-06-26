class Client:
    def __init__(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        args,
        device,
        model_trainer,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(
        self, client_idx, local_training_data, local_test_data, local_sample_number
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.infer_one_step(self.local_training_data, self.device, self.args,
                    move_to_gpu=True, model_train=True, clear_grad_bef_opt=True)
        compressed_grads, grad_indexes = self.model_trainer.get_model_grads()
        return compressed_grads, grad_indexes

    def get_model_bn(self):
        return self.model_trainer.get_model_bn()


    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
