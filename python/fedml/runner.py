from torch import nn

from .constants import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_NCCL,
    FEDML_TRAINING_PLATFORM_CROSS_SILO,
    FEDML_TRAINING_PLATFORM_CROSS_DEVICE,
    FEDML_SIMULATION_TYPE_MPI,
    FEDML_SIMULATION_TYPE_SP,
)
from .core import ClientTrainer, ServerAggregator, FedMLAlgorithmFlow


class FedMLRunner:
    def __init__(
        self,
        args,
        device,
        dataset,
        model,
        client_trainer: ClientTrainer = None,
        server_aggregator: ServerAggregator = None,
        algorithm_flow: FedMLAlgorithmFlow = None,
    ):
        if algorithm_flow is not None:
            self.runner = algorithm_flow
            return

        if args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION:
            init_runner_func = self._init_simulation_runner

        elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO:
            init_runner_func = self._init_cross_silo_runner

        elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_DEVICE:
            init_runner_func = self._init_cross_device_runner
        else:
            raise Exception("no such setting")

        self.runner = init_runner_func(
            args, device, dataset, model, client_trainer, server_aggregator
        )

    def _init_simulation_runner(
        self, args, device, dataset, model, client_trainer=None, server_aggregator=None
    ):
        if hasattr(args, "backend") and args.backend == FEDML_SIMULATION_TYPE_SP:
            from .simulation.simulator import SimulatorSingleProcess

            runner = SimulatorSingleProcess(
                args, device, dataset, model, client_trainer, server_aggregator
            )
        elif hasattr(args, "backend") and args.backend == FEDML_SIMULATION_TYPE_MPI:
            from .simulation.simulator import SimulatorMPI

            runner = SimulatorMPI(
                args, device, dataset, model, client_trainer, server_aggregator
            )
        elif hasattr(args, "backend") and args.backend == FEDML_SIMULATION_TYPE_NCCL:
            from .simulation.simulator import SimulatorNCCL

            runner = SimulatorNCCL(
                args, device, dataset, model, client_trainer, server_aggregator
            )
        else:
            raise Exception("not such backend {}".format(args.backend))

        return runner

    def _init_cross_silo_runner(
        self, args, device, dataset, model, client_trainer=None, server_aggregator=None
    ):
        if args.scenario == "horizontal":
            if args.role == "client":
                from .cross_silo import Client

                runner = Client(
                    args, device, dataset, model, client_trainer
                )
            elif args.role == "server":
                from .cross_silo import Server

                runner = Server(
                    args, device, dataset, model, server_aggregator
                )
            else:
                raise Exception("no such role")
        elif args.scenario == "hierarchical":
            if args.role == "client":
                from .cross_silo import Client

                runner = Client(
                    args, device, dataset, model, client_trainer
                )
            elif args.role == "server":
                from .cross_silo import Server

                runner = Server(
                    args, device, dataset, model, server_aggregator
                )
            else:
                raise Exception("no such role")
        else:
            raise Exception("no such setting")
        return runner

    def _init_cross_device_runner(
        self, args, device, dataset, model, client_trainer=None, server_aggregator=None
    ):
        if args.role == "server":
            from .cross_device import ServerMNN

            runner = ServerMNN(
                args, device, dataset, model, server_aggregator=server_aggregator
            )
        else:
            raise Exception(
                "Wrong program path: Python package only supports mobile server!"
            )
        return runner

    def run(self):
        self.runner.run()
