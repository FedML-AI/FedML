import random
from fedml.core.security.defense.SLSGD import SLSGD
from fedml.core.security.test.utils import (
    create_fake_model_list,
    create_fake_local_w_global_w,
)


def test_defense_option2():
    defense = SLSGD(trim_param_b=3, alpha=0.5, option_type=2)
    model_list = create_fake_model_list(20)
    random.shuffle(model_list)
    val = defense.defend(model_list)
    print(f"len={len(val)}, val={val}")


def test__sort_and_trim():
    defense = SLSGD(trim_param_b=3, alpha=0.5, option_type=2)
    model_list = create_fake_model_list(20)
    random.shuffle(model_list)
    print(f"len(model_list) = {len(model_list)}")
    processed_model_list = defense._sort_and_trim(model_list)
    print(
        f"len(trimed_list) = {len(processed_model_list)}, processed model list = {processed_model_list}"
    )


def test_aggregation():
    for alpha in [0, 0.5, 1]:
        avg_w, global_w = create_fake_local_w_global_w()
        print(f"avg_w = {avg_w}")
        print(f"global_w = {global_w}")
        print(
            f"alpha = {alpha}, aggregation = {SLSGD(trim_param_b=3, alpha=alpha, option_type=2).aggregate(avg_w, global_w)}"
        )


if __name__ == "__main__":
    test_defense_option2()
    test__sort_and_trim()
    test_aggregation()
