import copy
import logging

import torch
import wandb
from torch import nn

from fedml.core import mlops
from fedml.core.alg_frame.server_aggregator import ServerAggregator

from fedml.data.reddit.nlp import mask_tokens

class RedditAggregator(ServerAggregator):
    from transformers import (AdamW, AlbertTokenizer, AutoConfig,
                              AutoModelWithLMHead, AutoTokenizer,
                              MobileBertForPreTraining)

    tokenizer = AlbertTokenizer.from_pretrained(
        'albert-base-v2', do_lower_case=True)

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def _test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0, "perplexity_loss": 0.}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, _) in enumerate(test_data):
                # if args.mlm else (data, data)
                x, target = mask_tokens(
                    x, self.tokenizer, args, device=device)
                x = x.to(device)
                target = target.to(device)
                outputs = model(x, labels=target)
                loss = outputs[0]

                outputs = model(x, labels=target)

                loss = outputs[0]
                test_loss = loss.data.item()
                perplexity_loss = loss.data.item()

                acc = accuracy(
                    outputs[1].reshape(-1, outputs[1].shape[2]), target.reshape(-1), topk=(1, 5))

                correct = acc[0].item()
                top_5 = acc[1].item()

                metrics["test_correct"] += correct
                metrics["test_loss"] += test_loss * target.size(0)
                metrics["test_total"] += target.size(0)
                metrics["perplexity_loss"] += perplexity_loss * target.size(0)
        return metrics


    def test(self, test_data, device, args):
        # test data
        test_num_samples = []
        test_tot_corrects = []
        test_losses = []
        perplexity_losses = []

        metrics = self._test(test_data, device, args)

        test_tot_correct, test_num_sample, test_loss, perplexity_loss = (
            metrics["test_correct"],
            metrics["test_total"],
            metrics["test_loss"],
            metrics["perplexity_loss"],
        )
        test_tot_corrects.append(copy.deepcopy(test_tot_correct))
        test_num_samples.append(copy.deepcopy(test_num_sample))
        test_losses.append(copy.deepcopy(test_loss))
        perplexity_losses.append(copy.deepcopy(perplexity_loss))

        # test on test dataset
        test_acc = sum(test_tot_corrects) / sum(test_num_samples)
        test_loss = sum(test_losses) / sum(test_num_samples)
        perplexity_loss = sum(perplexity_losses) / sum(test_num_samples)

        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": args.round_idx})
            wandb.log({"Test/Loss": test_loss, "round": args.round_idx})
            wandb.log({"Test/Perplexity_Loss": perplexity_loss, "round": args.round_idx})

        mlops.log({"Test/Acc": test_acc, "round": args.round_idx})
        mlops.log({"Test/Loss": test_loss, "round": args.round_idx})
        mlops.log({"Test/Perplexity_Loss": perplexity_loss, "round": args.round_idx})

        stats = {"test_acc": test_acc, "test_loss": test_loss, "Test/Perplexity_Loss": perplexity_loss}
        logging.info(stats)

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        train_num_samples = []
        train_tot_corrects = []
        train_losses = []
        for client_idx in range(self.args.client_num_in_total):
            # train data
            metrics = self._test(train_data_local_dict[client_idx], device, args)
            train_tot_correct, train_num_sample, train_loss = (
                metrics["test_correct"],
                metrics["test_total"],
                metrics["test_loss"],
            )
            train_tot_corrects.append(copy.deepcopy(train_tot_correct))
            train_num_samples.append(copy.deepcopy(train_num_sample))
            train_losses.append(copy.deepcopy(train_loss))
            logging.info("client_idx = {}, metrics = {}".format(client_idx, metrics))

        # test on training dataset
        train_acc = sum(train_tot_corrects) / sum(train_num_samples)
        train_loss = sum(train_losses) / sum(train_num_samples)
        if self.args.enable_wandb:
            wandb.log({"Train/Acc": train_acc, "round": args.round_idx})
            wandb.log({"Train/Loss": train_loss, "round": args.round_idx})

        mlops.log({"Train/Acc": train_acc, "round": args.round_idx})
        mlops.log({"Train/Loss": train_loss, "round": args.round_idx})

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        logging.info(stats)

        return True


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)

        return res
