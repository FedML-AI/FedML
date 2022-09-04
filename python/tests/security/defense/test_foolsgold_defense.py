import argparse
from fedml.core.security.defense.foolsgold_defense import FoolsGoldDefense
from fedml.ml.aggregator.agg_operator import FedMLAggOperator
from utils import create_fake_model_list_MNIST, create_fake_model_list


def add_args(use_memory):
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file", "--cf", help="yaml configuration file", type=str, default="",
    )
    parser.add_argument("--federated_optimizer", type=str, default="FedAvg")
    parser.add_argument("--use_memory", type=bool, default=use_memory)
    args, unknown = parser.parse_known_args()
    return args


def test_fools_gold_score():
    model_list = create_fake_model_list(5)
    print(f"modellist len = {len(model_list)}")
    grads = [grad for num, grad in model_list]
    print(f"fools_gold_score={FoolsGoldDefense.fools_gold_score(grads)}")


def test_defense():
    config = add_args(use_memory=True)
    model_list = create_fake_model_list_MNIST(3)
    defense = FoolsGoldDefense(config)
    aggr_result = defense.run(model_list, base_aggregation_func=FedMLAggOperator.agg)
    print(f"result = {aggr_result}")


if __name__ == "__main__":
    test_fools_gold_score()
    test_defense()
