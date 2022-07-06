from fedml.core.security.attack.byzantine_attack import ByzantineAttack
from fedml.core.security.test.utils import create_fake_model_list, create_fake_model_list_MNIST


def test_get_malicious_client_idx():
    print(ByzantineAttack(byzantine_client_num=2, attack_mode="zero")._get_malicious_client_idx(10))
    print(ByzantineAttack(byzantine_client_num=10, attack_mode="zero")._get_malicious_client_idx(10))


def test__attack_zero_mode():
    local_w = create_fake_model_list(10)
    attack = ByzantineAttack(byzantine_client_num=2, attack_mode="zero")
    print(attack.attack(local_w, global_w=None))


def test__attack_random_mode():
    local_w = create_fake_model_list(10)
    attack = ByzantineAttack(byzantine_client_num=2, attack_mode="random")
    print(attack.attack(local_w, global_w=None))


def test__attack_test():
    local_w = create_fake_model_list_MNIST(10)
    print(ByzantineAttack._test_fedavg2(local_w))
    print("------------------")
    local_w = create_fake_model_list_MNIST(10)
    print(ByzantineAttack._test_fedavg(local_w))
    print("-------------")





if __name__ == '__main__':
    # test_get_malicious_client_idx()
    # print("---------")
    # test__attack_zero_mode()
    # print("---------")
    # test__attack_random_mode()

    test__attack_test()
