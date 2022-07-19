import random
from fedml.core.security.defense.slsgd_defense import SLSGDDefense
from fedml.core.security.test.aggregation.aggregation_functions import AggregationFunction
from fedml.core.security.test.utils import (
    create_fake_model_list,
    create_fake_local_w_global_w,
)


def test_defense_option2():
    defense = SLSGDDefense(trim_param_b=3, alpha=0.5, option_type=2)
    model_list = create_fake_model_list(20)
    random.shuffle(model_list)
    val = defense.run(model_list, base_aggregation_func=AggregationFunction.FedAVG, global_model=model_list[0][1])
    print(f"len={len(val)}, val={val}")


def test__sort_and_trim():
    defense = SLSGDDefense(trim_param_b=3, alpha=0.5, option_type=2)
    model_list = create_fake_model_list(20)
    random.shuffle(model_list)
    print(f"len(model_list) = {len(model_list)}")
    processed_model_list = defense._sort_and_trim(model_list)
    print(
        f"len(trimed_list) = {len(processed_model_list)}, processed model list = {processed_model_list}"
    )


def test_robustify_global_model():
    for alpha in [0, 0.5, 1]:
        model_list = create_fake_model_list(20)
        print(
            f"alpha = {alpha}, aggregation = {SLSGDDefense(trim_param_b=3, alpha=alpha, option_type=2).run(model_list, base_aggregation_func=AggregationFunction.FedAVG, global_model=model_list[0][1])} "
        )


if __name__ == "__main__":
    test_defense_option2()
    test__sort_and_trim()
    test_robustify_global_model()
