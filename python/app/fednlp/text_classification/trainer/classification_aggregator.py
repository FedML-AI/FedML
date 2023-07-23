import logging

import numpy as np
import torch
import wandb
from fedml import mlops
from fedml.core import ServerAggregator


# Trainer for MoleculeNet. The evaluation metric is ROC-AUC


class ClassificationAggregator(ServerAggregator):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        pass

    def _test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = torch.nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                x = batch[1].to(device)
                target = batch[4].to(device)
                # x = x.to(device)
                # target = target.to(device)
                pred = model(x)
                pred = pred[0]
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics

    def test_all(
        self, train_data_local_dict, test_data_local_dict, device, args=None
    ) -> bool:
        logging.info(f"----------test_on_the_server @ round {args.round_idx}--------")
        accuracy_list, loss_list, metric_list = [], [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            metrics = self._test(test_data, device, args)
            metric_list.append(metrics)
            accuracy_list.append(metrics["test_correct"] / metrics["test_total"])
            loss_list.append(metrics["test_loss"] / metrics["test_total"])
            logging.info(
                "Client {}, Test accuracy = {}".format(
                    client_idx, metrics["test_correct"] / metrics["test_total"]
                )
            )
        avg_accuracy = np.mean(np.array(accuracy_list))
        avg_loss = np.mean(np.array(loss_list))
        logging.info("Test Accuracy = {}".format(avg_accuracy))
        if self.args.enable_wandb:
            wandb.log({"Test/Acc": avg_accuracy, "round": self.args.round_idx})
            wandb.log({"Test/Loss": avg_loss, "round": self.args.round_idx})
        mlops.log({"round_idx": args.round_idx, "loss": avg_loss, "evaluation_result": avg_accuracy})
        return True
