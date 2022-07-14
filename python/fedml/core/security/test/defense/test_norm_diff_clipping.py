from fedml.core.security.defense.norm_diff_clipping_defense import NormDiffClippingDefense
from fedml.core.security.test.utils import (
    create_fake_vectors,
    create_fake_model_list_MNIST,
)


def test_norm_diff_clipping():
    defense = NormDiffClippingDefense(5.0)
    model_list = create_fake_model_list_MNIST(10)
    print(f"norm diff clipping result = {defense.defend(model_list, model_list[0][1])}")


def test__get_clipped_norm_diff():
    defense = NormDiffClippingDefense(5.0)
    local_v, global_v = create_fake_vectors()
    print(f"local_v = {local_v} \nglobal_v = {global_v}")
    print(f"clipped weight diff = {defense._get_clipped_norm_diff(local_v, global_v)}")


if __name__ == "__main__":
    test_norm_diff_clipping()
    test__get_clipped_norm_diff()
