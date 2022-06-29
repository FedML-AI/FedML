from fedml.core.security.defense.NormDiffClipping import NormDiffClipping
from fedml.core.security.test.utils import create_fake_vectors, create_fake_global_w_local_w_MNIST


def test_norm_diff_clipping():
    defense = NormDiffClipping(5.0)
    local_w, global_w = create_fake_global_w_local_w_MNIST()
    print(defense.defense(local_w, global_w))


def test__get_clipped_norm_diff():
    defense = NormDiffClipping(5.0)
    local_v, global_v = create_fake_vectors()
    print(f"local_v = {local_v} \nglobal_v = {global_v}")
    print(f"clipped weight diff = {defense._get_clipped_norm_diff(local_v, global_v)}")


if __name__ == '__main__':
    test_norm_diff_clipping()
    test__get_clipped_norm_diff()
