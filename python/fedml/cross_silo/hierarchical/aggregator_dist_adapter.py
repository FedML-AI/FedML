# from .FedAVGTrainer import FedAVGTrainer

# from ...standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
# from ...standalone.fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
# from ...standalone.fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
# from .utils import transform_list_to_tensor, post_complete_message_to_sweep_process
# from .message_define import MyMessage
# import logging
# import os
# import sys
# from .FedAVGAggregator import FedAVGAggregator


from .fedml_aggregator import FedMLAggregator
from .process_group_manager import ProcessGroupManager
from torch.nn.parallel import DistributedDataParallel as DDP
from .trainer.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .trainer.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from .trainer.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG


class AggregatorDistAdapter:
    def __init__(
        self,
        args,
        device,
        client_num,
        model,
        train_data_num,
        train_data_global,
        test_data_global,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        model_trainer,
    ):
        self.args = args

        only_gpu = args.using_gpu

        # if not self.args.is_mobile:
        self.process_group_manager = ProcessGroupManager(
            args.proc_rank_in_silo,
            args.n_proc_in_silo,
            args.pg_master_address,
            args.pg_master_port,
            only_gpu,
        )

        model.to(device)
        model = DDP(model, device_ids=[device] if only_gpu else None)

        if model_trainer is None:
            model_trainer = self.get_model_trainer(model, args)
        model_trainer.set_id(-1)

        # aggregator

        self.aggregator = self.get_aggregator(
            train_data_global,
            test_data_global,
            train_data_num,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            client_num,
            device,
            args,
            model_trainer,
        )

        self.device = device

    def get_aggregator(
        self,
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        client_num,
        device,
        args,
        model_trainer,
    ):
        worker_num = client_num
        return FedMLAggregator(
            train_data_global,
            test_data_global,
            train_data_num,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            worker_num,
            device,
            args,
            model_trainer,
        )

    def get_model_trainer(self, model, args):

        if args.dataset == "stackoverflow_lr":
            model_trainer = MyModelTrainerTAG(model, args, args.enable_cuda_rpc)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
            model_trainer = MyModelTrainerNWP(model, args, args.enable_cuda_rpc)
        else:  # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model, args, args.enable_cuda_rpc)
        return model_trainer

    def cleanup_pg(self):
        if not self.args.is_mobile:
            self.process_group_manager.cleanup()
