import logging
import torch
import torch.optim as optim

from fedml_api.distributed.fedavg.message_define import MyMessage
from fedml_core.distributed.client.client_manager import ClientMananger
from fedml_core.distributed.communication import Message

class SplitNN_client():
    def __init__(self, args):
        self.comm = args["comm"]
        self.model = args["model"]
        self.trainloader = args["trainloader"] 
        self.testloader = args["testloader"]
        self.rank = args["rank"]
        self.MAX_RANK = args["max_rank"]
        self.node_left = self.MAX_RANK if self.rank == 1 else self.rank - 1
        self.node_right = 1 if self.rank == self.MAX_RANK else self.rank + 1
        self.epoch_count = 0
        self.MAX_EPOCH_PER_NODE = args["epochs"]
        self.SERVER_RANK = args["server_rank"]
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,
                                   weight_decay=5e-4)

        self.trainloader = args["trainloader"]
        self.device = args["device"]

    def run(self):
        if self.rank == self.MAX_RANK:
            logging.info("sending semaphore from {} to {}".format(self.rank,
                                                                  self.node_right))
            self.comm.send("semaphore", dest=self.node_right)

        while(True):
            signal = self.comm.recv(source=self.node_left)

            if signal == "semaphore":
                logging.info("Starting training at node {}".format(self.rank))

            for batch_idx, (inputs, labels) in enumerate(self.trainloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                intermed_tensor = self.model(inputs)
                self.comm.send([intermed_tensor, labels], dest=self.SERVER_RANK)
                grads = self.comm.recv(source=self.SERVER_RANK)

                intermed_tensor.backward(grads)
                self.optimizer.step()

            logging.info("Epoch over at node {}".format(self.rank))
            del intermed_tensor, grads, inputs, labels
            torch.cuda.empty_cache()

            # Validation loss
            self.comm.send("validation", dest=self.SERVER_RANK)
            for batch_idx, (inputs, labels) in enumerate(self.testloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                intermed_tensor = self.model(inputs)
                self.comm.send([intermed_tensor, labels], dest=self.SERVER_RANK)

            del intermed_tensor, inputs, labels
            torch.cuda.empty_cache()

            self.epoch_count += 1
            self.comm.send("semaphore", dest=self.node_right)
            # self.comm.send(model.state_dict(), dest=node_right)
            if self.epoch_count == self.MAX_EPOCH_PER_NODE:
                if self.rank == self.MAX_RANK:
                    self.comm.send("over", dest=self.SERVER_RANK)
                self.comm.send("done", dest=self.SERVER_RANK)
                break
            self.comm.send("done", dest=self.SERVER_RANK)
 
