from .fedavg_server_optimizer import FedAvgServerOptimizer
from .fedprox_server_optimizer import FedProxServerOptimizer
# from .fedopt_server_optimizer import FedOptServerOptimizer
from .fednova_server_optimizer import FedNovaServerOptimizer
from .feddyn_server_optimizer import FedDynServerOptimizer
from .scaffold_server_optimizer import ScaffoldServerOptimizer
from .mime_server_optimizer import MimeServerOptimizer



def create_server_optimizer(args):
    if args.federated_optimizer in ["FedAvg", "FedAvg_seq"]:
        server_optimizer = FedAvgServerOptimizer(args)
    elif args.federated_optimizer == "FedProx":
        server_optimizer = FedProxServerOptimizer(args)
    elif args.federated_optimizer == "FedOpt":
        server_optimizer = FedOptServerOptimizer(args)
    elif args.federated_optimizer == "FedNova":
        server_optimizer = FedNovaServerOptimizer(args)
    elif args.federated_optimizer == "FedDyn":
        server_optimizer = FedDynServerOptimizer(args)
    elif args.federated_optimizer == "SCAFFOLD":
        server_optimizer = ScaffoldServerOptimizer(args)
    elif args.federated_optimizer == "Mime":
        server_optimizer = MimeServerOptimizer(args)
    else:  # default model trainer is for classification problem
        raise NotImplementedError
    return server_optimizer
