from .utils import transform_tensor_to_list, transform_list_to_tensor

from fedml.ml.ml_message import MLMessage

from fedml.core.alg_frame.params import Params

from fedml.ml.trainer.client_optimizer_creator import create_client_optimizer
from fedml.ml.trainer.local_cache import FedMLLocalCache

from fedml.core.compression.fedml_compression import FedMLCompression



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
        # self.trainer.set_id(client_index)
        # self.trainer.set_client_index(client_index)
        # self.trainer.set_server_result(server_result)
        self.trainer.set_id(client_index)
        self.trainer.client_index = client_index
        weights = server_result.get(MLMessage.MODEL_PARAMS)
        self.trainer.set_model_params(weights)
        self.server_result = server_result


    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]


    def load_client_status(self):
        if self.args.local_cache:
            if self.args.client_cache == "stateful":
                """
                In this mode, ClientTrainer is stateful during the whole training process.
                """
                client_status = self.client_status
            elif self.args.client_cache == "localhost":
                client_status = FedMLLocalCache.load(self.args, self.client_index)
            else:
                raise NotImplementedError
        else:
            client_status = {}
        return client_status

    def save_client_status(self, client_status={}):
        if self.args.local_cache:
            if client_status is None or len(client_status) == 0:
                return
            if self.args.client_cache == "stateful":
                """
                In this mode, ClientTrainer is stateful during the whole training process.
                """
                self.client_status = client_status
            elif self.args.client_cache == "localhost":
                FedMLLocalCache.save(self.args, self.client_index, client_status)
            else:
                raise NotImplementedError
        else:
            pass

    def train(self, round_idx=None):
        self.args.round_idx = round_idx

        client_status = self.load_client_status()
        FedMLCompression.get_instance().load_status(self.args, client_status)
        client_optimizer = create_client_optimizer(self.args)
        client_optimizer.load_status(self.args, client_status)
        client_optimizer.set_server_result(self.server_result)

        kwargs = {}
        kwargs["client_optimizer"] = client_optimizer
        self.trainer.train(self.train_local, self.device, self.args, **kwargs)

        client_result = Params()
        # weights_or_grads, params_to_server_optimizer = client_optimizer.end_local_training(args, self.client_index, model, train_data, device)
        other_result = client_optimizer.end_local_training(self.args, self.client_index,
                                                        self.trainer.model, self.train_local, self.device)
        client_result.add_dict(other_result)
        new_client_status = {"default": 0}
        new_client_status = client_optimizer.add_status(new_client_status)
        self.save_client_status(new_client_status)

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
