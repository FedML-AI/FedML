from .fedavg_client_optimizer import FedAvgClientOptimizer
from .fedprox_client_optimizer import FedProxClientOptimizer
from .fedopt_client_optimizer import FedOptClientOptimizer
from .fednova_client_optimizer import FedNovaClientOptimizer
from .feddyn_client_optimizer import FedDynClientOptimizer
from .scaffold_client_optimizer import ScaffoldClientOptimizer
from .mime_client_optimizer import MimeClientOptimizer
from .fedsgd_client_optimizer import FedSGDClientOptimizer
from .feddlc_client_optimizer import FedDLCClientOptimizer


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
        client_optimizer = ScaffoldClientOptimizer(args)
    elif args.federated_optimizer == "Mime":
        client_optimizer = MimeClientOptimizer(args)
    elif args.federated_optimizer == "FedSGD":
        client_optimizer = FedSGDClientOptimizer(args)
    elif args.federated_optimizer == "FedDLC":
        client_optimizer = FedDLCClientOptimizer(args)
    else:  # default model trainer is for classification problem
        raise NotImplementedError
    return client_optimizer
