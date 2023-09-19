from .lsa_fedml_aggregator import LightSecAggAggregator
from .lsa_fedml_client_manager import FedMLClientManager
from .lsa_fedml_server_manager import FedMLServerManager
from ..client.fedml_trainer import FedMLTrainer
from ...ml.trainer.trainer_creator import create_model_trainer


def FedML_LSA_Horizontal(
    args, client_rank, client_num, comm, device, dataset, model, model_trainer=None, preprocessed_sampling_lists=None,
):
    """
    Initialize and run the Federated Learning with LightSecAgg (LSA) in a horizontal setup.

    Args:
        args (object): Command-line arguments and configuration.
        client_rank (int): Rank or identifier of the current client (0 for the server).
        client_num (int): Total number of clients participating in the federated learning.
        comm (object): Communication backend for distributed training.
        device (object): The device on which the training will be performed (e.g., GPU or CPU).
        dataset (list): A list containing dataset-related information:
            - train_data_num (int): Number of samples in the global training dataset.
            - test_data_num (int): Number of samples in the global test dataset.
            - train_data_global (object): Global training dataset.
            - test_data_global (object): Global test dataset.
            - train_data_local_num_dict (dict): Dictionary mapping client indices to the number of local training samples.
            - train_data_local_dict (dict): Dictionary mapping client indices to their local training dataset.
            - test_data_local_dict (dict): Dictionary mapping client indices to their local test dataset.
            - class_num (int): Number of classes in the dataset.
        model (object): The federated learning model to be trained.
        model_trainer (object, optional): The model trainer responsible for training and testing. If not provided,
            it will be created based on the model and args.
        preprocessed_sampling_lists (list, optional): Preprocessed client sampling lists. If provided, the server will
            use these preprocessed sampling lists during initialization.

    Returns:
        None
    """
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset
    if client_rank == 0:
        init_server(
            args,
            device,
            comm,
            client_rank,
            client_num,
            model,
            train_data_num,
            train_data_global,
            test_data_global,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            model_trainer,
            preprocessed_sampling_lists,
        )
    else:
        init_client(
            args,
            device,
            comm,
            client_rank,
            client_num,
            model,
            train_data_num,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            model_trainer,
        )


def init_server(
    args,
    device,
    comm,
    client_rank,
    client_num,
    model,
    train_data_num,
    train_data_global,
    test_data_global,
    train_data_local_dict,
    test_data_local_dict,
    train_data_local_num_dict,
    model_trainer,
    preprocessed_sampling_lists=None,
):
    """
    Initialize the server for Federated Learning with LightSecAgg (LSA) in a horizontal setup.

    Args:
        args (object): Command-line arguments and configuration.
        device (object): The device on which the training will be performed (e.g., GPU or CPU).
        comm (object): Communication backend for distributed training.
        client_rank (int): Rank or identifier of the server (0 for the server).
        client_num (int): Total number of clients participating in the federated learning.
        model (object): The federated learning model to be trained.
        train_data_num (int): Number of samples in the global training dataset.
        train_data_global (object): Global training dataset.
        test_data_global (object): Global test dataset.
        train_data_local_dict (dict): Dictionary mapping client indices to their local training dataset.
        test_data_local_dict (dict): Dictionary mapping client indices to their local test dataset.
        train_data_local_num_dict (dict): Dictionary mapping client indices to the number of local training samples.
        model_trainer (object): The model trainer responsible for training and testing.
        preprocessed_sampling_lists (list, optional): Preprocessed client sampling lists. If provided, the server will
            use these preprocessed sampling lists during initialization.

    Returns:
        None
    """
    if model_trainer is None:
        model_trainer = create_model_trainer(model, args)
    model_trainer.set_id(0)

    # aggregator
    aggregator = LightSecAggAggregator(
        train_data_global,
        test_data_global,
        train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        client_num,
        device,
        args,
        model_trainer,
    )

    # start the distributed training
    backend = args.backend
    if preprocessed_sampling_lists is None:
        server_manager = FedMLServerManager(
            args, aggregator, comm, client_rank, client_num, backend)
    else:
        server_manager = FedMLServerManager(
            args,
            aggregator,
            comm,
            client_rank,
            client_num,
            backend,
            is_preprocessed=True,
            preprocessed_client_lists=preprocessed_sampling_lists,
        )
    # server_manager.send_init_msg()
    server_manager.run()


def init_client(
    args,
    device,
    comm,
    client_rank,
    client_num,
    model,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    model_trainer=None,
):
    """
    Initialize a client for Federated Learning with LightSecAgg (LSA) in a horizontal setup.

    Args:
        args (object): Command-line arguments and configuration.
        device (object): The device on which the training will be performed (e.g., GPU or CPU).
        comm (object): Communication backend for distributed training.
        client_rank (int): Rank or identifier of the current client.
        client_num (int): Total number of clients participating in the federated learning.
        model (object): The federated learning model to be trained.
        train_data_num (int): Number of samples in the global training dataset.
        train_data_local_num_dict (dict): Dictionary mapping client indices to the number of local training samples.
        train_data_local_dict (dict): Dictionary mapping client indices to their local training dataset.
        test_data_local_dict (dict): Dictionary mapping client indices to their local test dataset.
        model_trainer (object, optional): The model trainer responsible for training and testing. If not provided,
            it will be created based on the model and args.

    Returns:
        None
    """
    if model_trainer is None:
        model_trainer = create_model_trainer(model, args)
    model_trainer.set_id(client_rank)
    backend = args.backend
    trainer = FedMLTrainer(
        client_rank,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    )
    client_manager = FedMLClientManager(
        args, trainer, comm, client_rank, client_num, backend)
    client_manager.run()
