import logging

from .fedml_aggregator import FedMLAggregator
from .fedml_server_manager import FedMLServerManager
from ...ml.aggregator.aggregator_creator import create_server_aggregator


def fedavg_cross_device(args, process_id, worker_number, comm, device, test_dataloader, model, server_aggregator=None):
    logging.info("test_data_global.iter_number = {}".format(test_dataloader.iter_number))

    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number, model, test_dataloader, server_aggregator)


def init_server(args, device, comm, rank, size, model, test_dataloader, aggregator):
    if aggregator is None:
        aggregator = create_server_aggregator(model, args)
    aggregator.set_id(-1)

    td_id = id(test_dataloader)
    logging.info("test_dataloader = {}".format(td_id))
    logging.info("test_data_global.iter_number = {}".format(test_dataloader.iter_number))

    worker_num = size
    aggregator = FedMLAggregator(test_dataloader, worker_num, device, args, aggregator)

    # start the distributed training
    backend = args.backend
    server_manager = FedMLServerManager(args, aggregator, comm, rank, size, backend)
    if not args.using_mlops:
        server_manager.start_train()
    server_manager.run()
