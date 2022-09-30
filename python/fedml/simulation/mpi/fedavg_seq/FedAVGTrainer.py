from .utils import transform_tensor_to_list


class FedAVGTrainer(object):
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

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def get_lr(self, progress):
        # This aims to make a float step_size work.
        if self.args.lr_schedule == "StepLR":
            exp_num = progress / self.args.lr_step_size
            lr = self.args.learning_rate * (self.args.lr_decay_rate**exp_num)
        elif self.args.lr_schedule == "MultiStepLR":
            index = 0
            for milestone in self.args.lr_milestones:
                if progress < milestone:
                    break
                else:
                    index += 1
            lr = self.args.learning_rate * (self.args.lr_decay_rate**index)
        elif self.args.lr_schedule == "None":
            lr = self.args.learning_rate
        else:
            raise NotImplementedError
        return lr

    def train(self, round_idx=None):
        self.args.round_idx = round_idx
        # lr = self.get_lr(round_idx)
        # self.trainer.train(self.train_local, self.device, self.args, lr=lr)
        self.trainer.train(self.train_local, self.device, self.args)
        weights = self.trainer.get_model_params()

        return weights, self.local_sample_number

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
