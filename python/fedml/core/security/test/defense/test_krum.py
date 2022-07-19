from fedml.core.security.defense.krum_defense import KrumDefense
from fedml.core.security.test.aggregation.aggregation_functions import AggregationFunction
from fedml.core.security.test.utils import create_fake_model_list


def test_defense():
    model_list = create_fake_model_list(20)
    print(f"test krum")
    krum = KrumDefense(byzantine_client_num=2, multi=False)
    filtered_model_list = krum.run(model_list, base_aggregation_func=AggregationFunction.FedAVG)
    print(f"filtered_model_list={filtered_model_list}")

    print(f"test multi-krum")
    krum = KrumDefense(byzantine_client_num=2, multi=True)
    filtered_model_list = krum.run(model_list, base_aggregation_func=AggregationFunction.FedAVG)
    print(f"filtered_model_list={filtered_model_list}")


if __name__ == "__main__":
    test_defense()
