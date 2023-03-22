from mpi4py import MPI

from .guest_manager import GuestManager
from .guest_trainer import GuestTrainer
from .host_manager import HostManager
from .host_trainer import HostTrainer


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_VFL_distributed(
    process_id,
    worker_number,
    comm,
    args,
    device,
    guest_data,
    guest_model,
    host_data,
    host_model,
):
    if process_id == 0:
        init_guest_worker(
            args, comm, process_id, worker_number, device, guest_data, guest_model
        )
    else:
        init_host_worker(
            args, comm, process_id, worker_number, device, host_data, host_model
        )


def init_guest_worker(args, comm, process_id, size, device, guest_data, guest_model):
    Xa_train, y_train, Xa_test, y_test = guest_data
    model_feature_extractor, model_classifier = guest_model

    client_num = size - 1
    guest_trainer = GuestTrainer(
        client_num,
        device,
        Xa_train,
        y_train,
        Xa_test,
        y_test,
        model_feature_extractor,
        model_classifier,
        args,
    )

    server_manager = GuestManager(args, comm, process_id, size, guest_trainer)
    server_manager.run()


def init_host_worker(args, comm, process_id, size, device, host_data, host_model):
    X_train, X_test = host_data
    model_feature_extractor, model_classifier = host_model

    client_ID = process_id - 1
    trainer = HostTrainer(
        client_ID,
        device,
        X_train,
        X_test,
        model_feature_extractor,
        model_classifier,
        args,
    )

    client_manager = HostManager(args, comm, process_id, size, trainer)
    client_manager.run()
