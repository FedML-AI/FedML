import argparse
import logging

from fedml.core.security.defense.wbc_defense import WbcDefense
from utils import create_fake_model_list
from fedml.ml.aggregator.agg_operator import FedMLAggOperator

def add_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument(
        "--yaml_config_file", "--cf", help="yaml configuration file", type=str, default="",
    )
    parser.add_argument("--federated_optimizer", type=str, default="FedAvg")
    parser.add_argument("--client_idx", type=int, default=0)
    parser.add_argument("--batch_idx", type=int, default=1)
    parser.add_argument("--option_type", type=int, default=2)
    args, unknown = parser.parse_known_args()
    return args

def test_defense_wbc():
    args = add_args()
    model_list = create_fake_model_list(10)
    extra_auxiliary_info = create_fake_model_list(10)
    logging.info(f"test wbc")
    wbc = WbcDefense(args)
    filtered_model_list = wbc.run(
        model_list, FedMLAggOperator.agg, extra_auxiliary_info
    )
    logging.info(f"filtered_model_list={filtered_model_list}")


if __name__ == "__main__":
    test_defense_wbc()
