import argparse
from fedml.core.security.defense.crfl_defense import CRFLDefense
from fedml.ml.aggregator.agg_operator import FedMLAggOperator
from utils import create_fake_model_list_MNIST


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--federated_optimizer", type=str, default="FedAvg")
    args, unknown = parser.parse_known_args()
    return args


def test_defense():
    config = add_args()
    model_list = create_fake_model_list_MNIST(30)
    defense = CRFLDefense(config)
    global_model = defense.defend_on_aggregation(
        model_list, base_aggregation_func=FedMLAggOperator.agg
    )
    print(f"global_model = {global_model}")
    new_global_model = defense.defend_after_aggregation(global_model)
    print(f"global model after adding noise = {new_global_model}")


if __name__ == "__main__":
    test_defense()
