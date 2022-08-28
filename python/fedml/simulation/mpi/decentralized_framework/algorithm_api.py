import logging

from .decentralized_worker import DecentralizedWorker
from .decentralized_worker_manager import DecentralizedWorkerManager
from ....core.distributed.topology.symmetric_topology_manager import SymmetricTopologyManager


def FedML_Decentralized_Demo_distributed(args, process_id, worker_number, comm):
    # initialize the topology (ring)
    tpmgr = SymmetricTopologyManager(worker_number, 2)
    tpmgr.generate_topology()
    logging.info(tpmgr.topology)

    # initialize the decentralized trainer (worker)
    worker_index = process_id
    trainer = DecentralizedWorker(worker_index, tpmgr)

    client_manager = DecentralizedWorkerManager(args, comm, process_id, worker_number, trainer, tpmgr)
    client_manager.run()
