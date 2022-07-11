from .fedml_hierarchical_api import FedML_Hierarchical


class Server:
    def __init__(self, args, device, dataset, model, model_trainer=None):
        # Set inra-silo argiments
        args.rank_in_node = 0
        args.n_proc_per_node = 1
        args.n_proc_in_silo = 1
        args.proc_rank_in_silo = 0
        if args.federated_optimizer == "FedAvg":
            self.fl_trainer = FedML_Hierarchical(
                args,
                0,
                args.worker_num,
                None,
                device,
                dataset,
                model,
                model_trainer=model_trainer,
                preprocessed_sampling_lists=None,
            )

        else:
            raise Exception("Exception")

    def run(self):
        pass
