from fedml.core.security.attack.byzantine_attack import get_malicious_client_idx, ByzantineAttack
from fedml.core.security.test.utils import create_fake_model_list


def test_get_malicious_client_idx():
    print(get_malicious_client_idx(10, 2))
    print(get_malicious_client_idx(10, 10))


def test__attack_zero_mode():
    local_w = create_fake_model_list(10)
    # print(f"localw = {local_w}")
    # v = vectorize_weight(local_w)
    byzantine_idxs = get_malicious_client_idx(10, 2)
    print(f"byzantine client idx: {byzantine_idxs}")
    print(ByzantineAttack._attack_zero_mode(local_w, byzantine_idxs))


def test__attack_test():
    local_w = create_fake_model_list(10)
    print(ByzantineAttack._attack_test(local_w))



if __name__ == '__main__':
    # test_get_malicious_client_idx()
    test__attack_zero_mode()
    # test__attack_test()