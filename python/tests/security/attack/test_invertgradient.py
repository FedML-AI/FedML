import torch
from python.fedml.core.security.attack.invert_gradient_attack import loss_steps, InvertAttack, Classification
from python.fedml.core.security.common.attack_defense_data_loader import AttackDefenseDataLoader

import argparse


def add_config_untrained_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument("--signed", type=bool, default=True)
    parser.add_argument("--boxed", type=bool, default=True)
    parser.add_argument("--cost_fn", type=str, default="sim", help="Cost function to use")
    parser.add_argument("--indices", type=str, default="top10")
    parser.add_argument("--weights", type=str, default="equal")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--optim", type=str, default="adam", help="Optimization algorithm")
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=1)  # default: 24_000
    parser.add_argument("--total_variation", type=float, default=1e-6, help="Total variation")
    parser.add_argument("--init", type=str, default="randn")
    parser.add_argument("--filter", type=str, default="median")
    parser.add_argument("--lr_decay", type=bool, default=True)
    parser.add_argument("--scoring_choice", type=str, default="loss")
    parser.add_argument("--model_type", type=str, default="resnet18")
    parser.add_argument("--use_updates", type=bool, default=True)
    parser.add_argument("--num_images", type=int, default=1, help="batch_size in local training")

    args = parser.parse_args()
    return args


original_config_untrained = dict(
    signed=True, boxed=True, cost_fn="sim", indices="top10", weights="equal",
    lr=0.1, optim="adam", restarts=1, max_iterations=1,  # default: 4800
    total_variation=1e-6, init="randn", filter="median", lr_decay=True,
    scoring_choice="loss", model_type="resnet18", use_updates=True
)


def add_config_trained_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument("--signed", type=bool, default=True)
    parser.add_argument("--boxed", type=bool, default=True)
    parser.add_argument("--cost_fn", type=str, default="sim", help="Cost function to use")
    parser.add_argument("--indices", type=str, default="top10")
    parser.add_argument("--weights", type=str, default="equal")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--optim", type=str, default="adam", help="Optimization algorithm")
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=1)  # default: 24_000
    parser.add_argument("--total_variation", type=float, default=1e-2, help="Total variation")
    parser.add_argument("--init", type=str, default="randn")
    parser.add_argument("--filter", type=str, default="none")
    parser.add_argument("--lr_decay", type=bool, default=True)
    parser.add_argument("--scoring_choice", type=str, default="loss")
    parser.add_argument("--model_type", type=str, default="resnet18")
    parser.add_argument("--use_updates", type=bool, default=True)
    parser.add_argument("--num_images", type=int, default=1, help="batch_size in local training")

    args = parser.parse_args()
    return args


def add_config_untrained_one_img_args():
    parser = argparse.ArgumentParser(description="FedML")
    parser.add_argument("--signed", type=bool, default=True)
    parser.add_argument("--boxed", type=bool, default=True)
    parser.add_argument("--cost_fn", type=str, default="sim", help="Cost function to use")
    parser.add_argument("--indices", type=str, default="def")
    parser.add_argument("--weights", type=str, default="equal")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--optim", type=str, default="adam", help="Optimization algorithm")
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=1)  # default: 24_000
    parser.add_argument("--total_variation", type=float, default=1e-6, help="Total variation")
    parser.add_argument("--init", type=str, default="randn")
    parser.add_argument("--filter", type=str, default="none")
    parser.add_argument("--lr_decay", type=bool, default=True)
    parser.add_argument("--scoring_choice", type=str, default="loss")
    parser.add_argument("--model_type", type=str, default="resnet18")
    parser.add_argument("--use_updates", type=bool, default=True)
    parser.add_argument("--num_images", type=int, default=1, help="batch_size in local training")

    args = parser.parse_args()
    return args


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
    config = add_config_untrained_args()
    model = InvertAttack.get_model(config.model_type)  # torchvision.models.resnet18(pretrained=False)
    model.eval()
    refs, local_w = create_fake_input(0, 1, dataset, model, Classification(), False)
    attack = InvertAttack(args=config)
    # attack_client_idx = 0
    attack.reconstruct_data_using_a_model(a_gradient=local_w, extra_auxiliary_info=refs)


def test__attack_invertgradient_trained_oneimage():
    dataset = construct_dataloaders()
    config = add_config_trained_args()
    model = InvertAttack.get_model(config.model_type)  # torchvision.models.resnet18(pretrained=False)
    model.eval()
    refs, local_w = create_fake_input(0, 1, dataset, model, Classification(), False)
    attack = InvertAttack(args=config)
    # attack_client_idx = 0
    attack.reconstruct_data_using_a_model(a_gradient=local_w, extra_auxiliary_info=refs)


#
#
def test__attack_invertgradient_untrained_multiimage():
    dataset = construct_dataloaders()
    config = add_config_untrained_args()
    model = InvertAttack.get_model(config.model_type)  # torchvision.models.resnet18(pretrained=False)
    model.eval()
    refs, local_w = create_fake_input(0, 10, dataset, model, Classification(), False)
    attack = InvertAttack(args=config)
    attack.attack_model(raw_client_grad_list=local_w, extra_auxiliary_info=refs)


def test__attack_invertweight_untrained_oneimage():
    dataset = construct_dataloaders()
    config = add_config_untrained_one_img_args()
    model = InvertAttack.get_model(config.model_type)  # torchvision.models.resnet18(pretrained=False)
    model.eval()
    refs, local_w = create_fake_input(0, 1, dataset, model, Classification(), True)
    refs = (refs, config)
    attack = InvertAttack(args=config)
    attack.attack_model(raw_client_grad_list=local_w, extra_auxiliary_info=refs)


if __name__ == "__main__":
    dataset = construct_dataloaders()
    test__attack_invertgradient_untrained_oneimage()
    # test__attack_invertgradient_trained_oneimage()
    # test__attack_invertgradient_untrained_multiimage()
    # test__attack_invertweight_untrained_oneimage()
