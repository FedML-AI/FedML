from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from .FedAVGTrainer import FedAVGTrainer
import torch
import time

from ...standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from ...standalone.fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from ...standalone.fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
from .process_group_manager import ProcessGroupManager
from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process
from .message_define import MyMessage
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.getcwd(), "../../../../FedML")))

try:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.communication.utils import log_round_start, log_round_end
except ImportError:
    from fedml_core.distributed.client.client_manager import ClientManager
    from fedml_core.distributed.communication.message import Message
    from fedml_core.distributed.communication.utils import log_round_start, log_round_end


class DistWorker:
    def __init__(
        self,
        args,
        device,
        silo_rank,
        model,
        train_data_num,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer=None,
    ):

        only_gpu = bool(args.gpu_mapping_file)

        self.process_group_manager = ProcessGroupManager(
            args.silo_proc_rank, args.silo_proc_num, args.pg_master_address, args.pg_master_port, only_gpu
        )

        if not args.is_mobile:
            model.to(device)
            model = DDP(model, device_ids=[device] if only_gpu else None)


        client_index = silo_rank - 1
        if model_trainer is None:
            model_trainer = self.get_model_trainer(model, args)
        model_trainer.set_id(client_index)
        logging.info("Initiating Trainer")
        trainer = self.get_trainer(
            client_index,
            train_data_local_dict,
            train_data_local_num_dict,
            test_data_local_dict,
            train_data_num,
            device,
            args,
            model_trainer,
        )
        self.client_index = client_index
        self.silo_rank = silo_rank
        self.device = device
        self.trainer = trainer
        self.args = args

    def get_trainer(
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
        return FedAVGTrainer(
            client_index,
            train_data_local_dict,
            train_data_local_num_dict,
            test_data_local_dict,
            train_data_num,
            device,
            args,
            model_trainer,
        )

    def get_model_trainer(self, model, args):

        if args.dataset == "stackoverflow_lr":
            model_trainer = MyModelTrainerTAG(model, args.enable_cuda_rpc)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
            model_trainer = MyModelTrainerNWP(model, args.enable_cuda_rpc)
        else:  # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model, args.enable_cuda_rpc)
        return model_trainer

    def train(self, round_idx):

        log_round_start(self.silo_rank, round_idx)

        dist.barrier()
        weights, local_sample_num = self.trainer.train(round_idx)
        return weights, local_sample_num

    def update_model(self, model_params):
        self.trainer.update_model(model_params)

    def update_dataset(self, client_index=None):
        _client_index = client_index or self.client_index
        self.trainer.update_dataset(int(_client_index))

    def cleanup_pg(self):
        logging.info(
            "Cleaningup process group for client %s in silo %s" % (
                self.args.silo_proc_rank, self.args.silo_rank)
        )
        self.process_group_manager.cleanup()
