import logging
import traceback

from ..constants import (
    FEDML_FEDERATED_OPTIMIZER_BASE_FRAMEWORK,
    FEDML_FEDERATED_OPTIMIZER_FEDAVG,
    FEDML_FEDERATED_OPTIMIZER_FEDOPT,
    FEDML_FEDERATED_OPTIMIZER_FEDOPT_SEQ,
    FEDML_FEDERATED_OPTIMIZER_FEDNOVA,
    FEDML_FEDERATED_OPTIMIZER_FEDPROX,
    FEDML_FEDERATED_OPTIMIZER_CLASSICAL_VFL,
    FEDML_FEDERATED_OPTIMIZER_SPLIT_NN,
    FEDML_FEDERATED_OPTIMIZER_DECENTRALIZED_FL,
    FEDML_FEDERATED_OPTIMIZER_FEDGAN,
    FEDML_FEDERATED_OPTIMIZER_FEDAVG_SEQ,
    FEDML_FEDERATED_OPTIMIZER_FEDGKT,
    FEDML_FEDERATED_OPTIMIZER_FEDNAS,
    FEDML_FEDERATED_OPTIMIZER_FEDSEG,
    FEDML_FEDERATED_OPTIMIZER_HIERACHICAL_FL,
    FEDML_FEDERATED_OPTIMIZER_TURBO_AGGREGATE,
    FEDML_FEDERATED_OPTIMIZER_FEDSGD,
    FEDML_FEDERATED_OPTIMIZER_ASYNC_FEDAVG,
)
from ..core import ClientTrainer, ServerAggregator


class SimulatorSingleProcess:
    def __init__(self, args, device, dataset, model, client_trainer=None, server_aggregator=None):
        from .sp.classical_vertical_fl.vfl_api import VflFedAvgAPI
        from .sp.fedavg import FedAvgAPI
        from .sp.fednova.fednova_trainer import FedNovaTrainer
        from .sp.fedopt.fedopt_api import FedOptAPI
        from .sp.hierarchical_fl.trainer import HierachicalTrainer
        from .sp.turboaggregate.TA_trainer import TurboAggregateTrainer
        from .sp.fedsgd.fedsgd_api import FedSGDAPI

        if args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDAVG:
            self.fl_trainer = FedAvgAPI(args, device, dataset, model)
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDOPT:
            self.fl_trainer = FedOptAPI(args, device, dataset, model)
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDNOVA:
            self.fl_trainer = FedNovaTrainer(dataset, model, device, args)
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_HIERACHICAL_FL:
            self.fl_trainer = HierachicalTrainer(args, device, dataset, model)
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_TURBO_AGGREGATE:
            self.fl_trainer = TurboAggregateTrainer(dataset, model, device, args)
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_CLASSICAL_VFL:
            self.fl_trainer = VflFedAvgAPI(args, device, dataset, model)
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDSGD:
            self.fl_trainer = FedSGDAPI(args, device, dataset, model)

        # elif args.fl_trainer == FEDML_FEDERATED_OPTIMIZER_DECENTRALIZED_FL:
        #     self.fl_trainer = FedML_decentralized_fl()
        else:
            raise Exception("Exception")

    def run(self):
        self.fl_trainer.train()


class SimulatorMPI:
    def __init__(
        self,
        args,
        device,
        dataset,
        model,
        client_trainer: ClientTrainer = None,
        server_aggregator: ServerAggregator = None,
    ):
        from .mpi.base_framework.algorithm_api import FedML_Base_distributed
        from .mpi.decentralized_framework.algorithm_api import FedML_Decentralized_Demo_distributed
        from .mpi.fedavg.FedAvgAPI import FedML_FedAvg_distributed
        from .mpi.fedgkt.FedGKTAPI import FedML_FedGKT_distributed
        from .mpi.fednas.FedNASAPI import FedML_FedNAS_distributed
        from .mpi.fedopt.FedOptAPI import FedML_FedOpt_distributed
        from .mpi.fedopt_seq.FedOptSeqAPI import FedML_FedOptSeq_distributed
        from .mpi.fedprox.FedProxAPI import FedML_FedProx_distributed
        from .mpi.split_nn.SplitNNAPI import SplitNN_distributed
        from .mpi.fedgan.FedGanAPI import FedML_FedGan_distributed
        from .mpi.fedavg_seq.FedAvgSeqAPI import FedML_FedAvgSeq_distributed
        from .mpi.async_fedavg.AsyncFedAvgSeqAPI import FedML_Async_distributed

        if args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDAVG:
            self.simulator = FedML_FedAvg_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                client_trainer=client_trainer,
                server_aggregator=server_aggregator,
            )
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDAVG_SEQ:
            self.simulator = FedML_FedAvgSeq_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                client_trainer=client_trainer,
                server_aggregator=server_aggregator,
            )
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_BASE_FRAMEWORK:
            self.simulator = FedML_Base_distributed(args, args.process_id, args.worker_num, args.comm)
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDOPT:
            self.simulator = FedML_FedOpt_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                client_trainer=client_trainer,
                server_aggregator=server_aggregator,
            )
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDOPT_SEQ:
            self.simulator = FedML_FedOptSeq_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                client_trainer=client_trainer,
                server_aggregator=server_aggregator,
            )
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDPROX:
            self.simulator = FedML_FedProx_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                client_trainer=client_trainer,
                server_aggregator=server_aggregator,
            )
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_CLASSICAL_VFL:
            pass
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_SPLIT_NN:
            self.simulator = SplitNN_distributed(
                args.process_id, args.worker_num, device, args.comm, model, dataset=dataset, args=args,
            )
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_DECENTRALIZED_FL:
            self.simulator = FedML_Decentralized_Demo_distributed(args, args.process_id, args.worker_num, args.comm)
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDGAN:
            self.simulator = FedML_FedGan_distributed(
                args.process_id, args.worker_num, device, args.comm, model, args, dataset,
            )
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDGKT:
            self.simulator = FedML_FedGKT_distributed(
                args.process_id, args.worker_num, device, args.comm, model, dataset, args,
            )
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDNAS:
            self.simulator = FedML_FedNAS_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                client_trainer=client_trainer,
                server_aggregator=server_aggregator,
            )
        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_ASYNC_FEDAVG:
            self.simulator = FedML_Async_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                model_trainer=client_trainer,
                preprocessed_sampling_lists=None,
            )

        elif args.federated_optimizer == FEDML_FEDERATED_OPTIMIZER_FEDSEG:
            pass
        elif args.fl_trainer == FEDML_FEDERATED_OPTIMIZER_FEDGAN:
            self.fl_trainer = FedML_FedGan_distributed(
                args, args.process_id, args.worker_num, device, args.comm, model, dataset
            )
        else:
            raise Exception("Exception")

    def run(self):
        try:
            self.simulator.train()
        except Exception as e:
            logging.info("traceback.format_exc():\n%s" % traceback.format_exc())
            from mpi4py import MPI

            MPI.COMM_WORLD.Abort()


class SimulatorNCCL:
    def __init__(self, args, device, dataset, model, client_trainer=None, server_aggregator=None):
        from .nccl.fedavg.FedAvgAPI import FedML_FedAvg_NCCL

        if args.federated_optimizer == "FedAvg":
            self.simulator = FedML_FedAvg_NCCL(
                args, args.process_id, args.worker_num, args.comm, device, dataset, model, model_trainer=client_trainer,
            )
        else:
            raise Exception("Exception")

    def run(self):
        self.simulator.train()
