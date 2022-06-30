import logging
import traceback

from ..constants import (
    FedML_FEDERATED_OPTIMIZER_BASE_FRAMEWORK,
    FedML_FEDERATED_OPTIMIZER_FEDAVG,
    FedML_FEDERATED_OPTIMIZER_FEDOPT,
    FedML_FEDERATED_OPTIMIZER_FEDNOVA,
    FedML_FEDERATED_OPTIMIZER_FEDPROX,
    FedML_FEDERATED_OPTIMIZER_CLASSICAL_VFL,
    FedML_FEDERATED_OPTIMIZER_SPLIT_NN,
    FedML_FEDERATED_OPTIMIZER_DECENTRALIZED_FL,
    FedML_FEDERATED_OPTIMIZER_FEDGAN,
    FedML_FEDERATED_OPTIMIZER_FEDAVG_ROBUST,
    FedML_FEDERATED_OPTIMIZER_FEDGKT,
    FedML_FEDERATED_OPTIMIZER_FEDNAS,
    FedML_FEDERATED_OPTIMIZER_FEDSEG,
    FedML_FEDERATED_OPTIMIZER_HIERACHICAL_FL,
    FedML_FEDERATED_OPTIMIZER_TURBO_AGGREGATE,
    FedML_FEDERATED_OPTIMIZER_FEDSGD,
)


class SimulatorSingleProcess:
    def __init__(self, args, device, dataset, model):
        from .sp.classical_vertical_fl.vfl_api import VflFedAvgAPI
        from .sp.fedavg import FedAvgAPI
        from .sp.fednova.fednova_trainer import FedNovaTrainer
        from .sp.fedopt.fedopt_api import FedOptAPI
        from .sp.hierarchical_fl.trainer import HierachicalTrainer
        from .sp.turboaggregate.TA_trainer import TurboAggregateTrainer
        from .sp.fedsgd.fedsgd_api import FedSGDAPI

        if args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDAVG:
            self.fl_trainer = FedAvgAPI(args, device, dataset, model)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDOPT:
            self.fl_trainer = FedOptAPI(args, device, dataset, model)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDNOVA:
            self.fl_trainer = FedNovaTrainer(dataset, model, device, args)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_HIERACHICAL_FL:
            self.fl_trainer = HierachicalTrainer(args, device, dataset, model)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_TURBO_AGGREGATE:
            self.fl_trainer = TurboAggregateTrainer(dataset, model, device, args)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_CLASSICAL_VFL:
            self.fl_trainer = VflFedAvgAPI(dataset, model, device, args)
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDSGD:
            self.fl_trainer = FedSGDAPI(args, device, dataset, model)

        # elif args.fl_trainer == FedML_FEDERATED_OPTIMIZER_DECENTRALIZED_FL:
        #     self.fl_trainer = FedML_decentralized_fl()
        else:
            raise Exception("Exception")

    def run(self):
        self.fl_trainer.train()


class SimulatorMPI:
    def __init__(self, args, device, dataset, model, model_trainer=None):
        from .mpi.base_framework.algorithm_api import FedML_Base_distributed
        from .mpi.decentralized_framework.algorithm_api import (
            FedML_Decentralized_Demo_distributed,
        )
        from .mpi.fedavg.FedAvgAPI import FedML_FedAvg_distributed
        from .mpi.fedavg_robust.FedAvgRobustAPI import FedML_FedAvgRobust_distributed
        from .mpi.fedgkt.FedGKTAPI import FedML_FedGKT_distributed
        from .mpi.fednas.FedNASAPI import FedML_FedNAS_distributed
        from .mpi.fedopt.FedOptAPI import FedML_FedOpt_distributed
        from .mpi.fedprox.FedProxAPI import FedML_FedProx_distributed
        from .mpi.split_nn.SplitNNAPI import SplitNN_distributed
        from .mpi.fedgan.FedGanAPI import FedML_FedGan_distributed

        if args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDAVG:
            self.simulator = FedML_FedAvg_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                model_trainer=model_trainer,
                preprocessed_sampling_lists=None,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_BASE_FRAMEWORK:
            self.simulator = FedML_Base_distributed(
                args, args.process_id, args.worker_num, args.comm
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDOPT:
            self.simulator = FedML_FedOpt_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                model_trainer=model_trainer,
                preprocessed_sampling_lists=None,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDPROX:
            self.simulator = FedML_FedProx_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                model_trainer=model_trainer,
                preprocessed_sampling_lists=None,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_CLASSICAL_VFL:
            pass
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_SPLIT_NN:
            self.simulator = SplitNN_distributed(
                args.process_id,
                args.worker_num,
                device,
                args.comm,
                model,
                dataset=dataset,
                args=args,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_DECENTRALIZED_FL:
            self.simulator = FedML_Decentralized_Demo_distributed(
                args, args.process_id, args.worker_num, args.comm
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDGAN:
            self.simulator = FedML_FedGan_distributed(
                args.process_id,
                args.worker_num,
                device,
                args.comm,
                model,
                args,
                dataset,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDAVG_ROBUST:
            self.simulator = FedML_FedAvgRobust_distributed(
                args,
                args.process_id,
                args.worker_num,
                device,
                args.comm,
                model,
                dataset,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDGKT:
            self.simulator = FedML_FedGKT_distributed(
                args.process_id,
                args.worker_num,
                device,
                args.comm,
                model,
                dataset,
                args,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDNAS:
            self.simulator = FedML_FedNAS_distributed(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                model_trainer=model_trainer,
                preprocessed_sampling_lists=None,
            )
        elif args.federated_optimizer == FedML_FEDERATED_OPTIMIZER_FEDSEG:
            pass
        elif args.fl_trainer == FedML_FEDERATED_OPTIMIZER_FEDAVG_ROBUST:
            self.fl_trainer = FedML_FedAvgRobust_distributed(
                args,
                args.process_id,
                args.worker_num,
                device,
                args.comm,
                model,
                dataset,
            )
        elif args.fl_trainer == FedML_FEDERATED_OPTIMIZER_FEDGAN:
            self.fl_trainer = FedML_FedGan_distributed(args, device, dataset, model)
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
    def __init__(self, args, device, dataset, model, model_trainer=None):
        from .nccl.fedavg.FedAvgAPI import FedML_FedAvg_NCCL

        if args.federated_optimizer == "FedAvg":
            self.simulator = FedML_FedAvg_NCCL(
                args,
                args.process_id,
                args.worker_num,
                args.comm,
                device,
                dataset,
                model,
                model_trainer=model_trainer,
            )
        else:
            raise Exception("Exception")

    def run(self):
        self.simulator.train()
