import argparse
import torch
from fedml.core.security.defense.foolsgold_defense import FoolsGoldDefense


def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file",
        "--cf",
        help="yaml configuration file",
        type=str,
        default="",
    )
    parser.add_argument("--federated_optimizer", type=str, default="FedAvg")
    args, unknown = parser.parse_known_args()
    return args


def test_defense():
    config = add_args()
    model = torch.hub.load("pytorch/vision:v0.10.0", "vgg11", pretrained=True).state_dict()
    model_list = [(100, model) for i in range(4)]

    print(f"model_list len = {len(model_list)}")
    defense = FoolsGoldDefense(config)
    aggr_result = defense.defend_before_aggregation(model_list)
    # print(f"result = {aggr_result}")


if __name__ == "__main__":
    test_defense()
