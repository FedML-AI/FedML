from .fedml_aggregator import FedMLAggregator
from .fedml_server_manager import FedMLServerManager
from ..client.trainer.trainer_creator import create_model_trainer


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
    model_trainer
):
    if model_trainer is None:
        model_trainer = create_model_trainer(args, model)
    model_trainer.set_id(0)

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
        model_trainer,
    )

    # start the distributed training
    backend = args.backend
    server_manager = FedMLServerManager(
        args, aggregator, comm, rank, worker_num, backend
    )
    server_manager.run()
