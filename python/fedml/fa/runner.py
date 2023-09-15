from fedml import FEDML_SIMULATION_TYPE_SP, FEDML_TRAINING_PLATFORM_SIMULATION, FEDML_TRAINING_PLATFORM_CROSS_SILO
from fedml.fa.simulation.sp.simulator import FASimulatorSingleProcess

class FARunner:
    """
    A class for running Federated Learning simulations.

    Args:
        args: The arguments for configuring the simulation.
        dataset: The dataset used for the simulation.
        client_trainer: The client trainer for training clients (optional).
        server_aggregator: The server aggregator for aggregating client updates (optional).

    Methods:
        run():
            Run the Federated Learning simulation.

    """
    def __init__(
            self,
            args,
            dataset,
            client_trainer=None,
            server_aggregator=None,
    ):
        """
        Initialize the FARunner with the provided arguments and components.

        Args:
            args: The arguments for configuring the simulation.
            dataset: The dataset used for the simulation.
            client_trainer: The client trainer for training clients (optional).
            server_aggregator: The server aggregator for aggregating client updates (optional).

        Raises:
            Exception: If an invalid training type is specified in the arguments.

        """
        if args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION:
            init_runner_func = self._init_simulation_runner
        elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO:
            init_runner_func = self._init_cross_silo_runner
        else:
            raise Exception("no such setting")

        self.runner = init_runner_func(
            args, dataset, client_trainer, server_aggregator
        )

    def _init_simulation_runner(
            self, args, dataset, client_analyzer=None, server_analyzer=None
    ):
        """
        Initialize a simulation runner based on the provided arguments.

        Args:
            args: The arguments for configuring the simulation.
            dataset: The dataset used for the simulation.
            client_analyzer: The client analyzer for analyzing client behavior (optional).
            server_analyzer: The server analyzer for analyzing server behavior (optional).

        Returns:
            FASimulatorSingleProcess: A simulation runner for single-process simulation.

        Raises:
            Exception: If an unsupported simulation backend is specified in the arguments.

        """
        if hasattr(args, "backend") and args.backend == FEDML_SIMULATION_TYPE_SP:
            runner = FASimulatorSingleProcess(args, dataset)
        else:
            raise Exception("not such backend {}".format(args.backend))

        return runner

    def _init_cross_silo_runner(self, args, dataset, client_analyzer=None, server_analyzer=None):
        """
        Initialize a cross-silo runner based on the provided arguments.

        Args:
            args: The arguments for configuring the simulation.
            dataset: The dataset used for the simulation.
            client_analyzer: The client analyzer for analyzing client behavior (optional).
            server_analyzer: The server analyzer for analyzing server behavior (optional).

        Returns:
            FACrossSiloClient or FACrossSiloServer: A cross-silo client or server runner.

        Raises:
            Exception: If an invalid role is specified in the arguments.

        """
        if args.role == "client":
            from fedml.fa.cross_silo.fa_client import FACrossSiloClient as Client
            runner = Client(args, dataset, client_analyzer)
        elif args.role == "server":
            from fedml.fa.cross_silo.fa_server import FACrossSiloServer as Server
            runner = Server(args, dataset, server_analyzer)
        else:
            raise Exception("no such role")

        return runner

    def run(self):
        """
        Run the Federated Learning simulation.

        """
        self.runner.run()
