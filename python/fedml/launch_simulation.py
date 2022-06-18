import fedml
from .constants import (
    FEDML_TRAINING_PLATFORM_SIMULATION,
    FEDML_SIMULATION_TYPE_NCCL,
    FEDML_SIMULATION_TYPE_SP,
    FEDML_SIMULATION_TYPE_MPI,
)


def run_simulation(backend=FEDML_SIMULATION_TYPE_SP):

    """FedML Parrot"""
    fedml._global_training_type = FEDML_TRAINING_PLATFORM_SIMULATION
    fedml._global_comm_backend = backend

    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset, output_dim = fedml.data.load(args)

    # load model
    model = fedml.model.create(args, output_dim)

    # start training
    if backend == FEDML_SIMULATION_TYPE_SP:
        from .simulation.simulator import SimulatorSingleProcess

        simulator = SimulatorSingleProcess(args, device, dataset, model)
    elif backend == FEDML_SIMULATION_TYPE_NCCL:
        from .simulation.simulator import SimulatorNCCL

        simulator = SimulatorNCCL(args, device, dataset, model)

    elif backend == FEDML_SIMULATION_TYPE_MPI:
        from .simulation.simulator import SimulatorMPI

        simulator = SimulatorMPI(args, device, dataset, model)
    else:
        raise Exception("no such simulator {}".format(backend))
    simulator.run()
