from fedml.core.security.defense.cclip_defense import CClipDefense
from fedml.core.security.test.aggregation.aggregation_functions import AggregationFunction
from fedml.core.security.test.utils import create_fake_model_list


def test_defense():
    client_grad_list = create_fake_model_list(20)
    cclip = CClipDefense(tau=10, bucket_size=3)
    result = cclip.run(AggregationFunction.FedAVG, client_grad_list)
    print(f"result = {result}")


if __name__ == "__main__":
    test_defense()
