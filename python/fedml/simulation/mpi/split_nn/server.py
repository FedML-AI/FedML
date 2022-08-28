import logging

import torch.nn as nn
import torch.optim as optim


class SplitNN_server:
    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["model"]
        self.MAX_RANK = args["max_rank"]
        self.init_params()

    def init_params(self):
        self.epoch = 0
        self.log_step = 50
        self.active_node = 1
        self.train_mode()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

    def reset_local_params(self):
        logging.info("reset_local_params")
        self.total = 0
        self.correct = 0
        self.val_loss = 0
        self.step = 0
        self.batch_idx = 0

    def train_mode(self):
        logging.info("train_mode")
        self.model.train()
        self.phase = "train"
        self.reset_local_params()

    def eval_mode(self):
        logging.info("eval_mode")
        self.model.eval()
        self.phase = "validation"
        self.reset_local_params()

    def forward_pass(self, acts, labels):
        logging.info("forward_pass")
        self.acts = acts
        self.optimizer.zero_grad()
        self.acts.retain_grad()
        logits = self.model(acts)
        _, predictions = logits.max(1)
        self.loss = self.criterion(logits, labels)  # pylint: disable=E1102
        self.total += labels.size(0)
        self.correct += predictions.eq(labels).sum().item()
        if self.step % self.log_step == 0 and self.phase == "train":
            acc = self.correct / self.total
            logging.info(
                "phase={} acc={} loss={} epoch={} and step={}".format(
                    "train", acc, self.loss.item(), self.epoch, self.step
                )
            )
        if self.phase == "validation":
            self.val_loss += self.loss.item()
        self.step += 1

    def backward_pass(self):
        logging.info("backward_pass")
        self.loss.backward()
        self.optimizer.step()
        return self.acts.grad

    def validation_over(self):
        logging.info("validation_over")
        # not precise estimation of validation loss
        self.val_loss /= self.step
        acc = self.correct / self.total
        logging.info(
            "phase={} acc={} loss={} epoch={} and step={}".format(self.phase, acc, self.val_loss, self.epoch, self.step)
        )

        self.epoch += 1
        self.active_node = (self.active_node % self.MAX_RANK) + 1
        self.train_mode()
        logging.info("current active client is {}".format(self.active_node))
