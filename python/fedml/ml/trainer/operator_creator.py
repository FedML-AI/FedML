from .fedavg_train_operator import FedAvgClientOperator
from .fedprox_train_operator import FedProxClientOperator
from .fedopt_train_operator import FedOptClientOperator
from .fednova_train_operator import FedNovaClientOperator
from .feddyn_train_operator import FedDynClientOperator
from .scaffold_train_operator import SCAFFOLDClientOperator
from .mime_train_operator import MimeClientOperator



def create_trainer_operator(args):
    if args.federated_optimizer in ["FedAvg", "FedAvg_seq"]:
        client_operator = FedAvgClientOperator(args)
    elif args.federated_optimizer == "FedProx":
        client_operator = FedProxClientOperator(args)
    elif args.federated_optimizer == "FedOpt":
        client_operator = FedOptClientOperator(args)
    elif args.federated_optimizer == "FedNova":
        client_operator = FedNovaClientOperator(args)
    elif args.federated_optimizer == "FedDyn":
        client_operator = FedDynClientOperator(args)
    elif args.federated_optimizer == "SCAFFOLD":
        client_operator = SCAFFOLDClientOperator(args)
    elif args.federated_optimizer == "Mime":
        client_operator = MimeClientOperator(args)
    else:  # default model trainer is for classification problem
        raise NotImplementedError
    return client_operator
