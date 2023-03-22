from fedml.core.security.attack.invert_gradient_attack import (
    InvertAttack,
    Classification,
    loss_steps,
)
from fedml.core.security.common.attack_defense_data_loader import AttackDefenseDataLoader

import torch
import torchvision

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

"""Dataloader setups."""


def construct_dataloaders():
    client_num = 3
    batch_size = 32
    dataset = AttackDefenseDataLoader.load_cifar10_data(client_num=client_num, batch_size=batch_size)
    return dataset


def create_fake_input(attack_client_idx, num_images, images, model, loss_fn, use_updates):
    img = images[6][attack_client_idx].dataset

    if num_images == 1:
        idx = 0
        image, label = img[idx]
        labels = torch.as_tensor((label,))
        ground_truth = image.unsqueeze(0)
    else:
        ground_truth, labels = [], []
        idx = 0
        while len(labels) < num_images:
            image, label = img[idx]
            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,)))
                ground_truth.append(image)
        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)

    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)

    if not use_updates:
        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
        print(f"Full gradient norm is {full_norm:e}.")
        return (ground_truth, labels), input_gradient
    else:
        local_lr = 1e-4
        local_steps = 5
        input_parameters = loss_steps(
            model, ground_truth, labels, lr=local_lr, local_steps=local_steps, use_updates=use_updates,
        )
        input_parameters = [p.detach() for p in input_parameters]
        return (ground_truth, labels), input_parameters


def test__attack_invertgradient_untrained_oneimage():
    dataset = construct_dataloaders()
    config = config_untrained
    model = torchvision.models.resnet18(pretrained=False)
    model.eval()
    refs, local_w = create_fake_input(0, 1, dataset, model, Classification(), False)
    refs = (refs, config)
    attack = InvertAttack(model=model, attack_client_idx=0)
    # this is a reconstruction attack, I think there should be a new abstract method (e.g. recon_data)rather than attack_model or poison_data
    attack.attack_model(raw_client_grad_list=local_w, extra_auxiliary_info=refs)


def test__attack_invertgradient_trained_oneimage():
    dataset = construct_dataloaders()
    config = config_trained
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    refs, local_w = create_fake_input(0, 1, dataset, model, Classification(), False)
    refs = (refs, config)
    attack = InvertAttack(model=model, attack_client_idx=0, trained_model=True)
    attack.attack_model(raw_client_grad_list=local_w, extra_auxiliary_info=refs)


def test__attack_invertgradient_untrained_multiimage():
    dataset = construct_dataloaders()
    config = config_untrained
    model = torchvision.models.resnet18(pretrained=False)
    model.eval()
    refs, local_w = create_fake_input(0, 10, dataset, model, Classification(), False)
    refs = (refs, config)
    attack = InvertAttack(model=model, attack_client_idx=0, num_images=10)
    attack.attack_model(raw_client_grad_list=local_w, extra_auxiliary_info=refs)


def test__attack_invertweight_untrained_oneimage():
    dataset = construct_dataloaders()
    config = config_untrained_weight
    model = torchvision.models.resnet18(pretrained=False)
    model.eval()
    refs, local_w = create_fake_input(0, 1, dataset, model, Classification(), True)
    refs = (refs, config)
    attack = InvertAttack(model=model, attack_client_idx=0, use_updates=True)
    attack.attack_model(raw_client_grad_list=local_w, extra_auxiliary_info=refs)


if __name__ == "__main__":
    dataset = construct_dataloaders()
    test__attack_invertgradient_untrained_oneimage()
    test__attack_invertgradient_trained_oneimage()
    test__attack_invertgradient_untrained_multiimage()
    test__attack_invertweight_untrained_oneimage()
