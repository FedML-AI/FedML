import logging

import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    # def __init__(self):
    #     super(ModelTrainer, self).__init__()
    #     self.netg = self.model.get_netg()
    #     self.netd = self.model.get_netd()

    def get_model_params(self):
        weights_d = self.model.get_netd().cpu().state_dict()
        weights_g = self.model.get_netg().cpu().state_dict()
        weights = {'netg': weights_g, 'netd': weights_d}
        return weights

    def set_model_params(self, model_parameters):
        self.model.get_netg().load_state_dict(model_parameters['netg'])
        self.model.get_netd().load_state_dict(model_parameters['netd'])

    def train(self, train_data, device, args):
        netg = self.model.get_netg()
        netd = self.model.get_netd()

        netg.to(device)
        netg.train()
        netd.to(device)
        netd.train()

        criterion = nn.BCELoss()
        optimizer_g = torch.optim.Adam(netg.parameters(), lr=args.lr)
        optimizer_d = torch.optim.Adam(netd.parameters(), lr=args.lr)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_d_loss = []
            batch_g_loss = []
            for batch_idx, (x, _) in enumerate(train_data):
                # logging.info(images.shape)
                x = x.to(device)
                real_labels = torch.ones(x.size(0), 1).to(device)
                fake_labels = torch.zeros(x.size(0), 1).to(device)
                optimizer_d.zero_grad()
                d_real_loss = criterion(netd(x), real_labels)
                noise = torch.randn(x.size(0), 100).to(device)
                d_fake_loss = criterion(netd(netg(noise)), fake_labels)
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                optimizer_d.step()

                noise = torch.randn(x.size(0), 100).to(device)
                optimizer_g.zero_grad()
                g_loss = criterion(netd(netg(noise)), real_labels)
                g_loss.backward()
                optimizer_g.step()

                batch_d_loss.append(d_loss.item())
                batch_g_loss.append(g_loss.item())
            if len(batch_g_loss) > 0:
                epoch_loss.append(sum(batch_g_loss) / len(batch_g_loss))
                logging.info('(Trainer_ID {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.id,
                                                                                              epoch,
                                                                                              sum(epoch_loss) / len(
                                                                                                  epoch_loss)))

    def test(self, test_data, device, args):
        pass
        # metrics = {
        #     'test_correct': 1,
        #     'test_loss': 1,
        #     'test_precision': 1,
        #     'test_recall': 1,
        #     'test_total': 1
        # }
        # return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
