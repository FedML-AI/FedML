import copy
import logging

import numpy as np
from torch import nn

from fedml.core import ClientTrainer
from fedml.model.nlp.model_args import ClassificationArgs
from .text_classification_utils import *


class MyModelTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, test_data=None):
        model_args = ClassificationArgs()
        model_args.model_name = args.model
        model_args.model_type = args.model_type
        # model_args.load(model_args.model_name)
        # model_args.num_labels = output_dim
        model_args.update_from_dict(
            {
                "fl_algorithm": args.federated_optimizer,
                "freeze_layers": args.freeze_layers,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "do_lower_case": args.do_lower_case,
                "manual_seed": args.random_seed,
                # for ignoring the cache features.
                "reprocess_input_data": args.reprocess_input_data,
                "overwrite_output_dir": True,
                "max_seq_length": args.max_seq_length,
                "train_batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "evaluate_during_training": False,  # Disabled for FedAvg.
                "evaluate_during_training_steps": args.evaluate_during_training_steps,
                "fp16": args.fp16,
                "data_file_path": args.data_file_path,
                "partition_file_path": args.partition_file_path,
                "partition_method": args.partition_method,
                "dataset": args.dataset,
                "output_dir": args.output_dir,
                "is_debug_mode": args.is_debug_mode,
                "fedprox_mu": args.fedprox_mu,
                "optimizer": args.client_optimizer,
            }
        )
        model = self.model

        model.to(device)
        model.train()
        tr_loss = 0
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        iteration_in_total = (
            len(train_data) // args.gradient_accumulation_steps * args.epochs
        )
        optimizer, scheduler = build_optimizer(model, iteration_in_total, model_args)
        if args.federated_optimizer == "FedProx":
            global_model = copy.deepcopy(model)
        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, batch in enumerate(train_data):
                x = batch[1].to(device)
                labels = batch[4].to(device)
                log_probs = model(x)
                log_probs = log_probs[0]
                loss = criterion(log_probs, labels)
                if args.federated_optimizer == "FedProx":
                    fed_prox_reg = 0.0
                    mu = args.fedprox_mu
                    for (p, g_p) in zip(model.parameters(), global_model.parameters()):
                        fed_prox_reg += (mu / 2) * torch.norm((p - g_p.data)) ** 2
                    loss += fed_prox_reg

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                # logging.info(
                #    "Update Epoch: {} for Client Index: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #        self.id,
                #        epoch,
                #        (batch_idx + 1) * args.batch_size,
                #        len(train_data) * args.batch_size,
                #        100.0 * (batch_idx + 1) / len(train_data),
                #        loss.item(),
                #    )
                # )
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    if args.clip_grad_norm:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    batch_loss.append(tr_loss)
                    tr_loss = 0
                    # if args.evaluate_during_training and (args.evaluate_during_training_steps > 0 and global_step % args.evaluate_during_training_steps == 0):
                    #    metrics = self.

                    # global_step += 1

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # optimizer.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )
            if args.evaluate_during_training and test_data is not None:
                metrics = self.test(test_data, device, args)
                logging.info(
                    "Client Index = {}\tEpoch: {}\tAccuracy: {:.6f}".format(
                        self.id, epoch, metrics["test_correct"] / metrics["test_total"]
                    )
                )

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                if args.model_class == "transformer":
                    x = batch[1].to(device)
                    target = batch[4].to(device)
                else:
                    x, target = batch[0].to(device), batch[1].to(device)
                # x = x.to(device)
                # target = target.to(device)
                pred = model(x)
                if args.model_class == "transformer":
                    pred = pred[0]
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics

    def test_on_the_server(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        logging.info("----------test_on_the_server--------")
        accuracy_list, metric_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            metrics = self.test(test_data, device, args)
            metric_list.append(metrics)
            accuracy_list.append(metrics["test_correct"] / metrics["test_total"])
            logging.info(
                "Client {}, Test accuracy = {}".format(
                    client_idx, metrics["test_correct"] / metrics["test_total"]
                )
            )
        avg_accuracy = np.mean(np.array(accuracy_list))
        logging.info("Test Accuracy = {}".format(avg_accuracy))
        return True
