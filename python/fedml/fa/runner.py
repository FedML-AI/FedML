from fedml import FEDML_SIMULATION_TYPE_SP, FEDML_TRAINING_PLATFORM_SIMULATION, FEDML_TRAINING_PLATFORM_CROSS_SILO
from fedml.fa.simulation.sp.simulator import FASimulatorSingleProcess


class FARunner:
    def __init__(
            self,
            args,
            dataset,
            client_trainer=None,
            server_aggregator=None,
    ):

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
        if hasattr(args, "backend") and args.backend == FEDML_SIMULATION_TYPE_SP:
            runner = FASimulatorSingleProcess(args, dataset)
        else:
            raise Exception("not such backend {}".format(args.backend))

        return runner

    def _init_cross_silo_runner(self, args, dataset, client_analyzer=None, server_analyzer=None):
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
        self.runner.run()
