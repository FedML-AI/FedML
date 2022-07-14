import logging

from .server_mnn.server_mnn_api import fedavg_cross_device


class ServerMNN:
    def __init__(self, args, device, test_dataloader, model, client_trainer=None, server_aggregator=None):
        if args.federated_optimizer == "FedAvg":
            logging.info(
                "test_data_global.iter_number = {}".format(test_dataloader.iter_number)
            )

            self.fl_trainer = fedavg_cross_device(
                args,
                0,
                args.worker_num,
                None,
                device,
                test_dataloader,
                model,
                model_trainer=client_trainer,
                preprocessed_sampling_lists=None,
            )
        else:
            raise Exception("Exception")

    def run(self):
        pass
