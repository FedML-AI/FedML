import logging

from .fedml_aggregator import FedMLAggregator
from .fedml_server_manager import FedMLServerManager
from ...ml.trainer.trainer_creator import create_model_trainer


def fedavg_cross_device(
    args,
    process_id,
    worker_number,
    comm,
    device,
    test_dataloader,
    model,
    model_trainer=None,
    preprocessed_sampling_lists=None,
):
    logging.info("test_data_global.iter_number = {}".format(test_dataloader.iter_number))

    if process_id == 0:
        init_server(
            args,
            device,
            comm,
            process_id,
            worker_number,
            model,
            test_dataloader,
            model_trainer,
            preprocessed_sampling_lists,
        )


def init_server(
    args, device, comm, rank, size, model, test_dataloader, model_trainer, preprocessed_sampling_lists=None,
):
    if model_trainer is None:
        model_trainer = create_model_trainer(args, model)
    model_trainer.set_id(-1)

    td_id = id(test_dataloader)
    logging.info("test_dataloader = {}".format(td_id))
    logging.info("test_data_global.iter_number = {}".format(test_dataloader.iter_number))

    worker_num = size
    aggregator = FedMLAggregator(test_dataloader, worker_num, device, args, model_trainer,)

    # start the distributed training
    backend = args.backend
    if preprocessed_sampling_lists is None:
        server_manager = FedMLServerManager(args, aggregator, comm, rank, size, backend)
    else:
        server_manager = FedMLServerManager(
            args,
            aggregator,
            comm,
            rank,
            size,
            backend,
            is_preprocessed=True,
            preprocessed_client_lists=preprocessed_sampling_lists,
        )
    if not args.using_mlops:
        server_manager.start_train()
    server_manager.run()
