from fedml.core.security.defense.robust_learning_rate_defense import (
    RobustLearningRateDefense,
)
from fedml.core.security.test.utils import create_fake_model_list_MNIST


def test__compute_robust_learning_rates():
    defense = RobustLearningRateDefense(robust_threshold=1)
    model_list = create_fake_model_list_MNIST(10)
    print(f"{defense.run(model_list, base_aggregation_func=None, global_model=None)}")


if __name__ == "__main__":
    test__compute_robust_learning_rates()
