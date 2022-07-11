# test using local directory
# import os, sys
# __file__ = "/Users/kai/Documents/FedML/python/fedml"
# sys.path.append(__file__)
# from core.security.attack.invert_gradient_attack import InvertAttack
from fedml.core.security.attack.invert_gradient_attack import InvertAttack

config_untrained = dict(
    signed=True,
    boxed=True,
    cost_fn="sim",
    indices="top10",
    weights="equal",
    lr=0.1,
    optim="adam",
    restarts=1,
    max_iterations=1,  # default: 24_000
    total_variation=1e-6,
    init="randn",
    filter="median",
    lr_decay=True,
    scoring_choice="loss",
)

config_trained = dict(
    signed=True,
    boxed=True,
    cost_fn="sim",
    indices="top10",
    weights="equal",
    lr=0.1,
    optim="adam",
    restarts=1,
    max_iterations=1,  # default: 24_000
    total_variation=1e-2,
    init="randn",
    filter="none",
    lr_decay=True,
    scoring_choice="loss",
)

config_untrained_weight = dict(
    signed=True,
    boxed=True,
    cost_fn="sim",
    indices="def",
    weights="equal",
    lr=0.1,
    optim="adam",
    restarts=1,
    max_iterations=1,  # default: 8_000
    total_variation=1e-6,
    init="randn",
    filter="none",
    lr_decay=True,
    scoring_choice="loss",
)


def test__attack_invertgradient_untrained_oneimage(config):
    # local_gradient -> which could be inferred via w = w - eta * g
    attack = InvertAttack()
    attack.attack(local_w=None, global_w=None, refs=config)


def test__attack_invertgradient_trained_oneimage(config):
    # local_gradient -> which could be inferred via w = w - eta * g
    attack = InvertAttack(trained_model=True)
    attack.attack(local_w=None, global_w=None, refs=config)


def test__attack_invertgradient_untrained_multiimage(config):
    # local_gradient -> which could be inferred via w = w - eta * g
    attack = InvertAttack(num_images=10)
    attack.attack(local_w=None, global_w=None, refs=config)


def test__attack_invertweight_untrained_oneimage(config):
    # local_gradient -> which could be inferred via w = w - eta * g
    attack = InvertAttack(use_updates=True)
    attack.attack(local_w=None, global_w=None, refs=config)


if __name__ == "__main__":
    test__attack_invertgradient_untrained_oneimage(config_untrained)
    test__attack_invertgradient_trained_oneimage(config_trained)
    test__attack_invertgradient_untrained_multiimage(config_untrained)
    test__attack_invertweight_untrained_oneimage(config_untrained_weight)
