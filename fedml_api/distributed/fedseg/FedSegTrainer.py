import logging

import torch
from torch import nn

from fedml_api.distributed.fedseg.utils import transform_tensor_to_list
from fedml_api.distributed.fedseg.utils import SegmentationLosses, LR_Scheduler, Saver



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
        self.criterion = SegmentationLosses().build_loss(mode=args.loss_type).to(self.device)                   # modified 
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_local))

        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr,
                                              weight_decay=self.args.wd, amsgrad=True)

        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']                                              # add args.start_epoch
            self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:                                                                     # add args.ft
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))


        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0



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

        epoch_loss = []
        for epoch in range(self.args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(self.train_local):
                # logging.info(images.shape)
                x, labels = x.to(self.device), labels.to(self.device)
                self.scheduler(self.optimizer, batch_idx, epoch)
                self.optimizer.zero_grad()
                log_probs = self.model(x)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(client {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_index,
                                                                epoch, sum(epoch_loss) / len(epoch_loss)))


        # if self.args.no_val:                            
        # save checkpoint every epoch
        is_best = False
        self.saver.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': self.best_pred,
        }, is_best)

        weights = self.model.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number
