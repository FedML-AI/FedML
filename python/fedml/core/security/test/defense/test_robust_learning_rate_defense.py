from fedml.core.security.defense.robust_learning_rate_defense import RobustLearningRateDefense
from fedml.core.security.test.utils import create_fake_model_list_MNIST
import logging


def test__compute_robust_learning_rates():
    defense = RobustLearningRateDefense(robust_threshold=1)
    model_list = create_fake_model_list_MNIST(10)
    logging.info(
        "result = {}".format(defense.aggregate(model_list))
    )


if __name__ == '__main__':
    test__compute_robust_learning_rates()