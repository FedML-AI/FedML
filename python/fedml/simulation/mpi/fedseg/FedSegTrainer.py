from .utils import transform_tensor_to_list


class FedSegTrainer(object):
    def __init__(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        train_data_num,
        test_data_local_dict,
        device,
        model,
        args,
        model_trainer,
    ):
        self.args = args
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]
        self.args.round_idx = 0

        self.device = device

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def train(self):

        self.trainer.train(self.train_local, self.device)
        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number

    def test(self):
        train_evaluation_metrics = None

        if self.args.round_idx and self.args.round_idx % self.args.evaluation_frequency == 0:
            train_evaluation_metrics = self.trainer.test(self.train_local, self.device)

        test_evaluation_metrics = self.trainer.test(self.test_local, self.device)
        self.args.round_idx += 1
        return train_evaluation_metrics, test_evaluation_metrics
