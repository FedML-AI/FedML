from fedml.core.security.defense.wbc_defense import WbcDefense
from fedml.core.security.test.aggregation.aggregation_functions import (
    AggregationFunction,
)
from fedml.core.security.test.utils import create_fake_model_list

import logging

logging.getLogger().setLevel(logging.INFO)


def test_defense_wbc():
    model_list = create_fake_model_list(10)
    extra_auxiliary_info = create_fake_model_list(10)
    logging.info(f"test wbc")
    wbc = WbcDefense(client_idx=0, batch_idx=1)
    filtered_model_list = wbc.run(
        model_list, AggregationFunction.FedAVG, extra_auxiliary_info
    )
    logging.info(f"filtered_model_list={filtered_model_list}")


if __name__ == "__main__":
    test_defense_wbc()
