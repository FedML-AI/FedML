import logging

import torch.optim as optim


class SplitNN_client:
    """
    SplitNNClient class represents a client in a Split Learning setup.

    Args:
        args (dict): Dictionary containing client-specific configuration.
    """
    def __init__(self, args):
        """
        Initialize a SplitNNClient instance.

        Args:
            args (dict): Dictionary containing client-specific configuration.
        """
        self.client_idx = args['client_index']
        self.comm = args["comm"]
        self.model = args["model"]
        client_index = self.client_idx

        self.trainloader = args["trainloader"][client_index]
        self.testloader = args["testloader"][client_index]
        self.rank = args["rank"]
        self.MAX_RANK = args["max_rank"]
        self.node_left = self.MAX_RANK if self.rank == 1 else self.rank - 1
        self.node_right = 1 if self.rank == self.MAX_RANK else self.rank + 1
        self.epoch_count = 0
        self.batch_idx = 0
        self.MAX_EPOCH_PER_NODE = args["epochs"]
        self.SERVER_RANK = args["server_rank"]
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        self.device = args["device"]

    def forward_pass(self):
        """
        Perform a forward pass through the model.

        Returns:
            tuple: Tuple containing model activations (outputs) and labels.
        """
        logging.info("forward_pass")
        inputs, labels = next(self.dataloader)
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()

        try:
            self.acts = self.model(inputs)
        except:
            print(inputs.size())
            import pdb
            pdb.set_trace()
        return self.acts, labels

    def backward_pass(self, grads):
        """
        Perform a backward pass and update model parameters.

        Args:
            grads: Gradients used for the backward pass.
        """
        logging.info("backward_pass")
        self.acts.backward(grads)
        self.optimizer.step()

    def eval_mode(self):
        """
        Switch the model to evaluation mode and prepare the test data loader.
        """ 
        logging.info("eval_mode")
        self.dataloader = iter(self.testloader)
        self.model.eval()

    def train_mode(self):
        """
        Switch the model to training mode and prepare the training data loader.
        """
        logging.info("train_mode")
        self.dataloader = iter(self.trainloader)
        self.model.train()
