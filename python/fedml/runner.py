import os
from os.path import expanduser

from torch import nn

from .constants import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_NCCL,
    FEDML_TRAINING_PLATFORM_CROSS_SILO,
    FEDML_TRAINING_PLATFORM_CROSS_DEVICE,
    FEDML_TRAINING_PLATFORM_CHEETAH,
    FEDML_TRAINING_PLATFORM_SERVING,
    FEDML_SIMULATION_TYPE_MPI,
    FEDML_SIMULATION_TYPE_SP,
)
from .core import ClientTrainer, ServerAggregator, FedMLAlgorithmFlow


class FedMLRunner:
    """
    The main runner for different Federated Learning scenarios.

    Args:
        args: The command line arguments.
        device: The device (CPU or GPU) to use for training.
        dataset: The dataset used for training.
        model: The model to be trained.
        client_trainer (ClientTrainer, optional): The client trainer for training clients. Defaults to None.
        server_aggregator (ServerAggregator, optional): The server aggregator for aggregating client updates. Defaults to None.
        algorithm_flow (FedMLAlgorithmFlow, optional): The pre-defined algorithm flow. Defaults to None.
    """
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

        elif args.training_type == FEDML_TRAINING_PLATFORM_CHEETAH:
            init_runner_func = self._init_cheetah_runner

        elif args.training_type == FEDML_TRAINING_PLATFORM_SERVING:
            init_runner_func = self._init_model_serving_runner

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
        """
        Initialize the runner for the simulation-based Federated Learning.

        Args:
            args: The command line arguments.
            device: The device (CPU or GPU) to use for training.
            dataset: The dataset used for training.
            model: The model to be trained.
            client_trainer (ClientTrainer, optional): The client trainer for training clients. Defaults to None.
            server_aggregator (ServerAggregator, optional): The server aggregator for aggregating client updates. Defaults to None.

        Returns:
            runner: The initialized simulation-based runner.
        """
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
        """
        Initialize the runner for the cross-silo Federated Learning.

        Args:
            args: The command line arguments.
            device: The device (CPU or GPU) to use for training.
            dataset: The dataset used for training.
            model: The model to be trained.
            client_trainer (ClientTrainer, optional): The client trainer for training clients. Defaults to None.
            server_aggregator (ServerAggregator, optional): The server aggregator for aggregating client updates. Defaults to None.

        Returns:
            runner: The initialized cross-silo runner.
        """
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

    def _init_cheetah_runner(
            self, args, device, dataset, model, client_trainer=None, server_aggregator=None
    ):
        """
        Initialize the runner for the Cheetah Federated Learning.

        Args:
            args: The command line arguments.
            device: The device (CPU or GPU) to use for training.
            dataset: The dataset used for training.
            model: The model to be trained.
            client_trainer (ClientTrainer, optional): The client trainer for training clients. Defaults to None.
            server_aggregator (ServerAggregator, optional): The server aggregator for aggregating client updates. Defaults to None.

        Returns:
            runner: The initialized Cheetah runner.
        """
        if args.role == "client":
            from .cheetah import Client

            runner = Client(
                args, device, dataset, model, client_trainer
            )
        elif args.role == "server":
            from .cheetah import Server

            runner = Server(
                args, device, dataset, model, server_aggregator
            )
        else:
            raise Exception("no such role")
        return runner

    def _init_model_serving_runner(
            self, args, device, dataset, model, client_trainer=None, server_aggregator=None
    ):
        """
        Initialize the runner for the model serving Federated Learning.

        Args:
            args: The command line arguments.
            device: The device (CPU or GPU) to use for training.
            dataset: The dataset used for training.
            model: The model to be trained.
            client_trainer (ClientTrainer, optional): The client trainer for training clients. Defaults to None.
            server_aggregator (ServerAggregator, optional): The server aggregator for aggregating client updates. Defaults to None.

        Returns:
            runner: The initialized model serving runner.
        """
        if args.role == "client":
            from .serving import Client

            runner = Client(
                args, device, dataset, model, client_trainer
            )
        elif args.role == "server":
            from .serving import Server

            runner = Server(
                args, device, dataset, model, server_aggregator
            )
        else:
            raise Exception("no such role")
        return runner

    def _init_cross_device_runner(
        self, args, device, dataset, model, client_trainer=None, server_aggregator=None
    ):
        """
        Initialize the runner for the cross-device Federated Learning.

        Args:
            args: The command line arguments.
            device: The device (CPU or GPU) to use for training.
            dataset: The dataset used for training.
            model: The model to be trained.
            client_trainer (ClientTrainer, optional): The client trainer for training clients. Defaults to None.
            server_aggregator (ServerAggregator, optional): The server aggregator for aggregating client updates. Defaults to None.

        Returns:
            runner: The initialized cross-device runner.
        """
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

    @staticmethod
    def log_runner_result():
        """
        Log the result of the runner to a file.

        This method creates a log file containing the process ID and saves it to the "fedml_trace" directory.
        """
        log_runner_result_dir = os.path.join(expanduser("~"), "fedml_trace")
        if not os.path.exists(log_runner_result_dir):
            os.makedirs(log_runner_result_dir, exist_ok=True)

        log_file_obj = open(os.path.join(log_runner_result_dir, str(os.getpid())), "w")
        log_file_obj.write("{}".format(str(os.getpid())))
        log_file_obj.close()

    def run(self):
        """
        Run the initialized Federated Learning runner.

        This method executes the Federated Learning process using the selected runner.
        """
        self.runner.run()
        FedMLRunner.log_runner_result()

