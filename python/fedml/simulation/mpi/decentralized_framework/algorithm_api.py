import logging

from .decentralized_worker import DecentralizedWorker
from .decentralized_worker_manager import DecentralizedWorkerManager
from ....core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager

def FedML_Decentralized_Demo_distributed(args, process_id, worker_number, comm):
    """
    Run the decentralized federated learning demo on a distributed system.

    This function initializes the topology (ring) for decentralized federated learning,
    initializes the decentralized worker (trainer), and runs the decentralized worker manager.

    Args:
        args: Configuration arguments.
        process_id: The unique ID of the current process.
        worker_number: The total number of workers in the distributed system.
        comm: MPI communication object for distributed communication.
    """
    # Initialize the topology (ring)
    tpmgr = SymmetricTopologyManager(worker_number, 2)
    tpmgr.generate_topology()
    logging.info(tpmgr.topology)

    # Initialize the decentralized trainer (worker)
    worker_index = process_id
    trainer = DecentralizedWorker(worker_index, tpmgr)

    # Initialize the decentralized worker manager
    client_manager = DecentralizedWorkerManager(args, comm, process_id, worker_number, trainer, tpmgr)
    client_manager.run()
