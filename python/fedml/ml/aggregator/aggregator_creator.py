from .default_aggregator import DefaultServerAggregator
from .my_server_aggregator_nwp import MyServerAggregatorNWP
from .my_server_aggregator_prediction import MyServerAggregatorTAGPred


def create_server_aggregator(model, args):
    if args.dataset == "stackoverflow_lr":
        aggregator = MyServerAggregatorTAGPred(model, args)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        aggregator = MyServerAggregatorNWP(model, args)
    else:
        aggregator = DefaultServerAggregator(model, args)
    return aggregator
