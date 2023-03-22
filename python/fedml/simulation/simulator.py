
from ..constants import (
    FedML_FEDERATED_OPTIMIZER_BASE_FRAMEWORK,
    FedML_FEDERATED_OPTIMIZER_FEDAVG,
    FedML_FEDERATED_OPTIMIZER_FEDOPT,
    FedML_FEDERATED_OPTIMIZER_FEDOPT_SEQ,
    FedML_FEDERATED_OPTIMIZER_FEDNOVA,
    FedML_FEDERATED_OPTIMIZER_FEDDYN,
    FedML_FEDERATED_OPTIMIZER_FEDPROX,
    FedML_FEDERATED_OPTIMIZER_CLASSICAL_VFL,
    FedML_FEDERATED_OPTIMIZER_SPLIT_NN,
    FedML_FEDERATED_OPTIMIZER_DECENTRALIZED_FL,
    FedML_FEDERATED_OPTIMIZER_FEDGAN,
    FedML_FEDERATED_OPTIMIZER_FEDAVG_SEQ,
    FedML_FEDERATED_OPTIMIZER_FEDGKT,
    FedML_FEDERATED_OPTIMIZER_FEDNAS,
    FedML_FEDERATED_OPTIMIZER_SCAFFOLD,
    FedML_FEDERATED_OPTIMIZER_MIME,
    FedML_FEDERATED_OPTIMIZER_FEDSEG,
    FedML_FEDERATED_OPTIMIZER_HIERACHICAL_FL,
    FedML_FEDERATED_OPTIMIZER_TURBO_AGGREGATE,
    FedML_FEDERATED_OPTIMIZER_ASYNC_FEDAVG,
)
from ..core import ClientTrainer, ServerAggregator


class SimulatorSingleProcess:
    def __init__(self, args, device, dataset, model, client_trainer=None, server_aggregator=None):
        from .sp.classical_vertical_fl.vfl_api import VflFedAvgAPI
        from .sp.fedavg import FedAvgAPI
        from .sp.fedprox.fedprox_trainer import FedProxTrainer
        from .sp.fednova.fednova_trainer import FedNovaTrainer
        from .sp.feddyn.feddyn_trainer import FedDynTrainer
        from .sp.scaffold.scaffold_trainer import ScaffoldTrainer
        from .sp.mime.mime_trainer import MimeTrainer
        from .sp.fedopt.fedopt_api import FedOptAPI
        from .sp.hierarchical_fl.trainer import HierarchicalTrainer
        from .sp.turboaggregate.TA_trainer import TurboAggregateTrainer

        if args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDAVG:
            self.fl_trainer = FedAvgAPI(args, device, dataset, model)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDOPT:
            self.fl_trainer = FedOptAPI(args, device, dataset, model)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDNOVA:
            self.fl_trainer = FedNovaTrainer(dataset, model, device, args)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDDYN:
            self.fl_trainer = FedDynTrainer(dataset, model, device, args)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDPROX:
            self.fl_trainer = FedProxTrainer(dataset, model, device, args)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_SCAFFOLD:
            self.fl_trainer = ScaffoldTrainer(dataset, model, device, args)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_MIME:
            self.fl_trainer = MimeTrainer(dataset, model, device, args)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_HIERACHICAL_FL:
            self.fl_trainer = HierarchicalTrainer(args, device, dataset, model)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_TURBO_AGGREGATE:
            self.fl_trainer = TurboAggregateTrainer(dataset, model, device, args)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_CLASSICAL_VFL:
            self.fl_trainer = VflFedAvgAPI(args, device, dataset, model)

        # elif args.fl_trainer == FedML_FEDERATED_OPTIMIZER_DECENTRALIZED_FL:
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
        from .mpi.fednova.FedNovaAPI import FedML_FedNova_distributed

        if args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDAVG:
            FedML_FedAvg_distributed(
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
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDAVG_SEQ:
            FedML_FedAvgSeq_distributed(
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
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_BASE_FRAMEWORK:
            FedML_Base_distributed(args, args.process_id, args.worker_num, args.comm)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDOPT:
            FedML_FedOpt_distributed(
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
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDOPT_SEQ:
            FedML_FedOptSeq_distributed(
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
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDPROX:
            FedML_FedProx_distributed(
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
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_CLASSICAL_VFL:
            pass
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_SPLIT_NN:
            SplitNN_distributed(
                args.process_id, args.worker_num, device, args.comm, model, dataset=dataset, args=args,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_DECENTRALIZED_FL:
            FedML_Decentralized_Demo_distributed(args, args.process_id, args.worker_num, args.comm)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDGAN:
            FedML_FedGan_distributed(
                args, args.process_id, args.worker_num, device, args.comm, model, dataset
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDGKT:
            FedML_FedGKT_distributed(
                args.process_id, args.worker_num, device, args.comm, model, dataset, args,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDNAS:
            FedML_FedNAS_distributed(
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
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_ASYNC_FEDAVG:
            FedML_Async_distributed(
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
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDNOVA:
            FedML_FedNova_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                client_trainer=client_trainer,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDSEG:
            pass
        elif args.fl_trainer == FedML_FEDERATED_OPTIMIZER_FEDGAN:
            FedML_FedGan_distributed(args, args.process_id, args.worker_num, device, args.comm, model, dataset)
        else:
            raise Exception("Exception")

    def run(self):
        pass


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
