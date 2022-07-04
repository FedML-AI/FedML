from fedml.constants import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_SP,
    FEDML_SIMULATION_TYPE_MPI,
    FEDML_SIMULATION_TYPE_NCCL,
    FEDML_TRAINING_PLATFORM_CROSS_SILO,
    FEDML_TRAINING_PLATFORM_CROSS_DEVICE,
)


class FedMLTrainer:
    def __init__(self, args, device, dataset, model, model_trainer=None):
        self.trainer = None
        if (
            args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION
            and hasattr(args, "backend")
            and args.backend == FEDML_SIMULATION_TYPE_MPI
        ):
            from fedml.simulation import SimulatorMPI

            self.trainer = SimulatorMPI(args, device, dataset, model)

        elif (
            args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION
            and hasattr(args, "backend")
            and args.backend == FEDML_SIMULATION_TYPE_SP
        ):
            from fedml.simulation import SimulatorSingleProcess as Simulator

            self.trainer = Simulator(args, device, dataset, model)
        elif (
            args.training_type == FEDML_TRAINING_PLATFORM_SIMULATION
            and hasattr(args, "backend")
            and args.backend == FEDML_SIMULATION_TYPE_NCCL
        ):
            from fedml.simulation import SimulatorNCCL

            self.trainer = SimulatorNCCL(args, device, dataset, model)

        elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_SILO:
            if not hasattr(args, "scenario"):
                args.scenario = "horizontal"
            if args.scenario == "horizontal":

                self.trainer = None

            elif args.scenario == "hierarchical":
                self.trainer = None

        elif args.training_type == FEDML_TRAINING_PLATFORM_CROSS_DEVICE:
            self.trainer = None
        else:
            raise Exception("no such trainer")

    def run(self):
        self.trainer.run()
