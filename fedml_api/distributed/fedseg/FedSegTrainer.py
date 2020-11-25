import logging, os, time

import torch
from torch import nn

from fedml_api.distributed.fedseg.utils import transform_tensor_to_list
from fedml_api.distributed.fedseg.utils import SegmentationLosses, LR_Scheduler



class FedSegTrainer(object):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict, train_data_num, device, model,
                 args):

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.device = device
        self.args = args
        self.model = model
        self.model.to(self.device)
        self.criterion = SegmentationLosses().build_loss(mode=self.args.loss_type)                   # modified 
        self.scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr, self.args.epochs, self.train_data_local_num_dict[client_index])

        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)


    def update_model(self, weights):
        # logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self):

        self.model.to(self.device)
        # change to train mode
        self.model.train()
        
        logging.info('Training client {0} for {1} Epochs'.format(self.client_index, self.args.epochs))
        epoch_loss = []
        
        for epoch in range(self.args.epochs):
            t = time.time()
            batch_loss = []

            logging.info('Client Id: {0}, Epoch: {1}'.format(self.client_index, epoch))

            for (batch_idx, batch) in enumerate(self.train_local):
                x, labels = batch['image'], batch['label']
                x, labels = x.to(self.device), labels.to(self.device)
                self.scheduler(self.optimizer, batch_idx, epoch)
                self.optimizer.zero_grad()
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels).to(self.device)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
                if (batch_idx % 500 == 0):
                    logging.info('Client Id: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.client_index, batch_idx, loss, (time.time()-t)/60))

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Client Id: {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_index,
                                                                epoch, sum(epoch_loss) / len(epoch_loss)))

            logging.info('Client Id: {0} Epoch: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.client_index, epoch, batch_loss[-1], (time.time()-t)/60))

        weights = self.model.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number
