from .fedavg_client_optimizer import FedAvgClientOptimizer
from .fedprox_client_optimizer import FedProxClientOptimizer
# from .fedopt_client_optimizer import FedOptClientOptimizer
# from .fednova_client_optimizer import FedNovaClientOptimizer
# from .feddyn_client_optimizer import FedDynClientOptimizer
# from .scaffold_client_optimizer import SCAFFOLDClientOptimizer
# from .mime_client_optimizer import MimeClientOptimizer



def create_client_optimizer(args):
    if args.federated_optimizer in ["FedAvg", "FedAvg_seq"]:
        client_optimizer = FedAvgClientOptimizer(args)
    elif args.federated_optimizer == "FedProx":
        client_optimizer = FedProxClientOptimizer(args)
    elif args.federated_optimizer == "FedOpt":
        client_optimizer = FedOptClientOptimizer(args)
    elif args.federated_optimizer == "FedNova":
        client_optimizer = FedNovaClientOptimizer(args)
    elif args.federated_optimizer == "FedDyn":
        client_optimizer = FedDynClientOptimizer(args)
    elif args.federated_optimizer == "SCAFFOLD":
        client_optimizer = SCAFFOLDClientOptimizer(args)
    elif args.federated_optimizer == "Mime":
        client_optimizer = MimeClientOptimizer(args)
    else:  # default model trainer is for classification problem
        raise NotImplementedError
    return client_optimizer
