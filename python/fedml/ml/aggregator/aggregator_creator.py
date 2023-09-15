from .default_aggregator import DefaultServerAggregator
from .my_server_aggregator_nwp import MyServerAggregatorNWP
from .my_server_aggregator_prediction import MyServerAggregatorTAGPred


def create_server_aggregator(model, args):
    """
    Create a server aggregator instance based on the selected dataset and configuration parameters.

    Args:
        model: The machine learning model to be used for aggregation.
        args: A dictionary containing training configuration parameters, including the dataset.

    Returns:
        ServerAggregator: An instance of a server aggregator class suitable for the specified dataset.
    """
    if args.dataset == "stackoverflow_lr":
        aggregator = MyServerAggregatorTAGPred(model, args)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        aggregator = MyServerAggregatorNWP(model, args)
    else:
        aggregator = DefaultServerAggregator(model, args)
    return aggregator
