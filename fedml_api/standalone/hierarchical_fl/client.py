import copy
import torch

from fedml_api.standalone.fedavg.client import Client

class Client(Client):

    def train(self, global_round_idx, group_round_idx, w):
        self.model.load_state_dict(w)
        self.model.to(self.device)

        if self.args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

        w_list = []
        for epoch in range(self.args.epochs):
            for x, labels in self.local_training_data:
                x, labels = x.to(self.device), labels.to(self.device)
                self.model.zero_grad()
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
            global_epoch = global_round_idx*self.args.group_comm_round*self.args.epochs + \
                            group_round_idx*self.args.epochs + epoch
            if global_epoch % self.args.frequency_of_the_test == 0 or epoch == self.args.epochs-1:
                w_list.append((global_epoch, copy.deepcopy(self.model.state_dict())))
        return w_list
