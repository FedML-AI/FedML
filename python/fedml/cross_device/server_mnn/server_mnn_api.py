from .fedml_aggregator import FedMLAggregator
from .fedml_server_manager import FedMLServerManager
from .trainer.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .trainer.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from .trainer.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG

import logging


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
    logging.info(
        "test_data_global.iter_number = {}".format(test_dataloader.iter_number)
    )

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
    args,
    device,
    comm,
    rank,
    size,
    model,
    test_dataloader,
    model_trainer,
    preprocessed_sampling_lists=None,
):
    if model_trainer is None:
        if args.dataset == "stackoverflow_lr":
            model_trainer = MyModelTrainerTAG(model)
        elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
            model_trainer = MyModelTrainerNWP(model)
        else:  # default model trainer is for classification problem
            model_trainer = MyModelTrainerCLS(model)
    model_trainer.set_id(-1)

    # aggregator

    td_id = id(test_dataloader)
    logging.info("test_dataloader = {}".format(td_id))
    logging.info(
        "test_data_global.iter_number = {}".format(test_dataloader.iter_number)
    )

    worker_num = size
    aggregator = FedMLAggregator(
        test_dataloader,
        worker_num,
        device,
        args,
        model_trainer,
    )

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
