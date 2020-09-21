import datetime
import logging

import torch.nn as nn
import torch.optim as optim


class SplitNN_server():
    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["model"]
        self.MAX_RANK = args["max_rank"]
        self.active_node = 1
        self.epoch = 0
        self.batch_idx = 0
        self.step = 0
        self.log_step = 50
        self.active_node = 1
        self.phase = "train"
        self.val_loss = 0
        self.total = 0
        self.correct = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,
                                   weight_decay=5e-4)
        self.criterion = nn.CrossEntropyLoss()

    def run(self, ):
        while (True):
            message = self.comm.recv(source=self.active_node)
            if message == "done":
                # not a precise estimate of validation loss
                self.val_loss /= self.step
                acc = self.correct / self.total
                logging.info("phase={} acc={} loss={} epoch={} and step={}"
                             .format(self.phase, acc, self.loss.item(), self.epoch, self.step))

                self.epoch += 1
                self.active_node = (self.active_node % self.MAX_RANK) + 1
                self.phase = "train"
                self.total = 0
                self.correct = 0
                self.val_loss = 0
                self.step = 0
                self.batch_idx = 0
                logging.info("current active client is {}".format(self.active_node))
            elif message == "over":
                logging.info("training over")
                break
            elif message == "validation":
                self.phase = "validation"
                self.step = 0
                self.total = 0
                self.correct = 0
            else:
                if self.phase == "train":
                    logging.debug("Server-Receive: client={}, index={}, time={}"
                                  .format(self.active_node, self.batch_idx,
                                          datetime.datetime.now()))
                    self.optimizer.zero_grad()
                input_tensor, labels = message
                input_tensor.retain_grad()
                logits = self.model(input_tensor)
                _, predictions = logits.max(1)

                loss = self.criterion(logits, labels)
                self.loss = loss
                self.total += labels.size(0)
                self.correct += predictions.eq(labels).sum().item()

                if self.phase == "train":
                    loss.backward()
                    self.optimizer.step()
                    self.comm.send(input_tensor.grad, dest=self.active_node)
                    self.batch_idx += 1

                self.step += 1
                if self.step % self.log_step == 0 and self.phase == "train":
                    acc = self.correct / self.total
                    logging.info("phase={} acc={} loss={} epoch={} and step={}"
                                 .format("train", acc, loss.item(), self.epoch, self.step))
                if self.phase == "validation":
                    self.val_loss += loss.item()
