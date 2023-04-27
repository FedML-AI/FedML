from .fedml_aggregator import FedMLAggregator
from .fedml_server_manager import FedMLServerManager
from ...ml.aggregator.aggregator_creator import create_server_aggregator


def init_server(
    args,
    device,
    comm,
    rank,
    worker_num,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_dict,
    test_data_local_dict,
    train_data_local_num_dict,
    server_aggregator,
):
    if server_aggregator is None:
        server_aggregator = create_server_aggregator(model, args)
    server_aggregator.set_id(0)

    # aggregator
    aggregator = FedMLAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        server_aggregator,
    )

    # start the distributed training
    backend = args.backend
    server_manager = FedMLServerManager(args, aggregator, comm, rank, worker_num, backend)
    server_manager.run()
