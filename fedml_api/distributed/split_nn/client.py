import torch.optim as optim


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
        self.batch_idx = 0
        self.MAX_EPOCH_PER_NODE = args["epochs"]
        self.SERVER_RANK = args["server_rank"]
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9,
                                   weight_decay=5e-4)

        self.trainloader = args["trainloader"]
        self.device = args["device"]

    def forward_pass(self):
        inputs, labels = next(self.dataloader)
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()

        self.acts = self.model(inputs)
        return self.acts, labels

    def backward_pass(self, grads):
        self.acts.backward(grads)
        self.optimizer.step()

    def eval_mode(self):
        self.dataloader = iter(self.testloader)
        self.model.eval()

    def train_mode(self):
        self.dataloader = iter(self.trainloader)
        self.model.train()
