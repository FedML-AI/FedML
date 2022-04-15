from torch.nn.parallel import DistributedDataParallel as DDP

from ...standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from ...standalone.fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from ...standalone.fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
from .process_group_manager import ProcessGroupManager
import os
import sys
from .FedAVGAggregator import FedAVGAggregator

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


class DistAggregator():
    def __init__(self, args,
                 device,
                 size,
                 model,
                 train_data_num,
                 train_data_global,
                 test_data_global,
                 train_data_local_dict,
                 test_data_local_dict,
                 train_data_local_num_dict,
                 model_trainer):
        self.args = args

        only_gpu = bool(args.gpu_mapping_file)

        if not self.args.is_mobile:
            self.process_group_manager = ProcessGroupManager(
                args.silo_proc_rank, args.silo_proc_num, args.pg_master_address, args.pg_master_port, only_gpu)
            model = DDP(model, device_ids=[device] if only_gpu else None)

            model.to(device)

        if model_trainer is None:
            if args.dataset == "stackoverflow_lr":
                model_trainer = MyModelTrainerTAG(model, args.enable_cuda_rpc)
            elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
                model_trainer = MyModelTrainerNWP(model, args.enable_cuda_rpc)
            else:  # default model trainer is for classification problem
                model_trainer = MyModelTrainerCLS(model, args.enable_cuda_rpc)
        model_trainer.set_id(-1)

        # aggregator

        self.aggregator = self.get_aggregator(
            train_data_global,
            test_data_global,
            train_data_num,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            size,
            device,
            args,
            model_trainer,
        )

        self.device = device

    def get_aggregator(self,
                       train_data_global,
                       test_data_global,
                       train_data_num,
                       train_data_local_dict,
                       test_data_local_dict,
                       train_data_local_num_dict,
                       size,
                       device,
                       args,
                       model_trainer):
        worker_num = size - 1
        return FedAVGAggregator(
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

    def cleanup_pg(self):
        if not self.args.is_mobile:
            self.process_group_manager.cleanup()
