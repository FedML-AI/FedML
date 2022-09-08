import argparse
from fedml.core.security.defense.robust_learning_rate_defense import (
    RobustLearningRateDefense,
)
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
    # default arguments
    parser.add_argument("--robust_threshold", type=int, default=1)
    args, unknown = parser.parse_known_args()
    return args


def test_robust_learning_rate_defense():
    defense = RobustLearningRateDefense(add_args())
    model_list = create_fake_model_list_MNIST(10)
    print(f"{defense.run(model_list)}")


if __name__ == "__main__":
    test_robust_learning_rate_defense()
