import copy

import fedml
from fedml.core import ContributionAssessorManager
import logging

import torch
import torch.nn as nn


def evaluation_func(model, test_data, device):
    model = model

    model.to(device)
    model.eval()

    metrics = {
        "test_correct": 0,
        "test_loss": 0,
        "test_precision": 0,
        "test_recall": 0,
        "test_total": 0,
    }

    """
    stackoverflow_lr is the task of multi-label classification
    please refer to following links for detailed explainations on cross-entropy and corresponding implementation of tff research:
    https://towardsdatascience.com/cross-entropy-for-classification-d98e7f974451
    https://github.com/google-research/federated/blob/49a43456aa5eaee3e1749855eed89c0087983541/optimization/stackoverflow_lr/federated_stackoverflow_lr.py#L131
    """
    if args.dataset == "stackoverflow_lr":
        criterion = nn.BCELoss(reduction="sum").to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(test_data):
            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            loss = criterion(pred, target)

            if args.dataset == "stackoverflow_lr":
                predicted = (pred > 0.5).int()
                correct = predicted.eq(target).sum(axis=-1).eq(target.size(1)).sum()
                true_positive = ((target * predicted) > 0.1).int().sum(axis=-1)
                precision = true_positive / (predicted.sum(axis=-1) + 1e-13)
                recall = true_positive / (target.sum(axis=-1) + 1e-13)
                metrics["test_precision"] += precision.sum().item()
                metrics["test_recall"] += recall.sum().item()
            else:
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

            metrics["test_correct"] += correct.item()
            metrics["test_loss"] += loss.item() * target.size(0)
            if len(target.size()) == 1:  #
                metrics["test_total"] += target.size(0)
            elif len(target.size()) == 2:  # for tasks of next word prediction
                metrics["test_total"] += target.size(0) * target.size(1)
    logging.info("test_correct = {}, test_total = {}".format(metrics["test_correct"], metrics["test_total"]))
    acc = float(metrics["test_correct"])/float(metrics["test_total"])
    return acc


if __name__ == "__main__":
    args = fedml.init()

    # load data
    dataset, output_dim = fedml.data.load(args)
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    # init device
    device = fedml.device.get_device(args)

    # load model
    model = fedml.model.create(args, output_dim)

    model_list_from_client_update = [copy.deepcopy(model) for i in range(args.client_num_per_round)]
    model_aggregated = copy.deepcopy(model_list_from_client_update[0])
    model_last_round = copy.deepcopy(model_list_from_client_update[1])
    acc_on_aggregated_model = 0.85
    val_dataloader = test_data_local_dict[1]

    contribution_assessor_mgr = ContributionAssessorManager(args)
    contribution_vector = contribution_assessor_mgr.run(
        model_list_from_client_update,
        model_aggregated,
        model_last_round,
        acc_on_aggregated_model,
        val_dataloader,
        evaluation_func,
        device,
    )
