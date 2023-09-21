import time

from fedml.data import split_data_for_dist_trainers
from ...constants import FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL
from ...core.mlops.mlops_profiler_event import MLOpsProfilerEvent


class FedMLTrainer(object):
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

        if args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
            self.train_data_local_dict = split_data_for_dist_trainers(train_data_local_dict, args.n_proc_in_silo)
        else:
            self.train_data_local_dict = train_data_local_dict

        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = None
        self.local_sample_number = None
        self.test_local = None

        self.device = device
        self.args = args
        self.args.device = device

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index

        if self.train_data_local_dict is not None:
            if self.args.scenario == FEDML_CROSS_SILO_SCENARIO_HIERARCHICAL:
                self.train_local = self.train_data_local_dict[client_index][self.args.proc_rank_in_silo]
            else:
                self.train_local = self.train_data_local_dict[client_index]
        else:
            self.train_local = None

        if self.train_data_local_num_dict is not None:
            self.local_sample_number = self.train_data_local_num_dict[client_index]
        else:
            self.local_sample_number = 0

        if self.test_data_local_dict is not None:
            self.test_local = self.test_data_local_dict[client_index]
        else:
            self.test_local = None

        self.trainer.update_dataset(self.train_local, self.test_local, self.local_sample_number)

    def train(self, round_idx=None):
        self.args.round_idx = round_idx
        tick = time.time()

        self.trainer.on_before_local_training(self.train_local, self.device, self.args)
        self.trainer.train(self.train_local, self.device, self.args)
        self.trainer.on_after_local_training(self.train_local, self.device, self.args)

        MLOpsProfilerEvent.log_to_wandb({"Train/Time": time.time() - tick, "round": round_idx})
        weights = self.trainer.get_model_params()
        # transform Tensor to list
        return weights, self.local_sample_number

    def test(self, round_idx=None):
        self.args.round_idx = round_idx
        if hasattr(self.trainer, "test"):
            self.trainer.test(self.test_local, self.device, self.args)