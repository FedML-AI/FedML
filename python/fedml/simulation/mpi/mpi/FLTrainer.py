from .utils import transform_tensor_to_list, transform_list_to_tensor

from fedml.ml.ml_message import MLMessage

class FLTrainer(object):
    def __init__(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    ):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.device = device
        self.args = args

    # def update_model(self, weights):
    #     self.trainer.set_model_params(weights)

    def update_trainer(self, client_index, server_result):
        self.trainer.set_id(client_index)
        self.trainer.set_client_index(client_index)
        weights = server_result[MLMessage.MODEL_PARAMS]
        if self.args.is_mobile == 1:
            model_params = transform_list_to_tensor(model_params)

        self.trainer.set_model_params(weights)
        self.params_to_client_optimizer = server_result[MLMessage.PARAMS_TO_CLIENT_OPTIMIZER]
        # self.trainer.set_params_to_client_optimizer(params_to_client_optimizer)


    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def train(self, round_idx=None):
        self.args.round_idx = round_idx
        # self.trainer.train(self.train_local, self.device, self.args)
        local_loss, params_to_server_optimizer = \
            self.trainer.train(self.train_local, self.device, self.args, 
                self.params_to_client_optimizer)

        weights = self.trainer.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)

        client_result = {}
        client_result[MLMessage.MODEL_PARAMS] = weights
        client_result[MLMessage.PARAMS_TO_SERVER_OPTIMIZER] = params_to_server_optimizer

        # return weights, self.local_sample_number
        return client_result, self.local_sample_number

    def test(self):
        # train data
        train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        train_tot_correct, train_num_sample, train_loss = (
            train_metrics["test_correct"],
            train_metrics["test_total"],
            train_metrics["test_loss"],
        )

        # test data
        test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        test_tot_correct, test_num_sample, test_loss = (
            test_metrics["test_correct"],
            test_metrics["test_total"],
            test_metrics["test_loss"],
        )

        return (
            train_tot_correct,
            train_loss,
            train_num_sample,
            test_tot_correct,
            test_loss,
            test_num_sample,
        )
