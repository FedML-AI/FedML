import logging, os, time
from os import path
import torch
from torch import nn
import numpy as np

from fedml_api.distributed.fedseg.utils import transform_tensor_to_list, SegmentationLosses, Evaluator, LR_Scheduler, EvaluationMetricsKeeper



class FedSegTrainer(object):
    def __init__(self, client_index, train_data_local_dict, train_data_local_num_dict,
                 train_data_num, test_data_local_dict, device, model, n_class, args):

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.test_data_local_dict = test_data_local_dict
        self.all_train_data_num = train_data_num
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]
        
        self.device = device
        self.args = args
        self.model = model
        self.model.to(self.device)
        self.criterion = SegmentationLosses().build_loss(mode=self.args.loss_type) 
        self.evaluator = Evaluator(n_class)
        self.scheduler = LR_Scheduler(self.args.lr_scheduler, self.args.lr, self.args.epochs, self.train_data_local_num_dict[client_index])

        # Add momentum if needed
        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr)
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
        self.test_local = self.test_data_local_dict[client_index]

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

    def test(self):

        # Train Data
        train_evaluation_metrics = self._infer(self.train_local)

        # Test Data        
        test_evaluation_metrics = self._infer(self.test_local)

        # Test on training dataset
        return train_evaluation_metrics, test_evaluation_metrics


    def _infer(self, test_data):
        time_start_test_per_batch = time.time()
        self.model.eval()
        self.model.to(self.device)
        self.evaluator.reset()

        test_acc = test_acc_class = test_mIoU = test_FWIoU = test_loss = test_total = 0.
        criterion = SegmentationLosses().build_loss(mode=self.args.loss_type)

        with torch.no_grad():
            for (batch_idx, batch) in enumerate(test_data):
                x, target = batch['image'], batch['label']
                x, target = x.to(self.device), target.to('device')
                output = self.model(x)
                loss = criterion(output, target).to(self.device)
                test_loss += loss.item()
                test_total += target.size(0)
                pred = output.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis = 1)
                self.evaluator.add_batch(target, pred)
                time_end_test_per_batch = time.time()
                logging.info("time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))

        # Evaluation Metrics (Averaged over number of samples)
        test_acc = self.evaluator.Pixel_Accuracy()
        test_acc_class = self.evaluator.Pixel_Accuracy_Class()
        test_mIoU = self.evaluator.Mean_Intersection_over_Union()
        test_FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        test_loss = test_loss / test_total
        
        return EvaluationMetricsKeeper(test_acc, test_acc_class, test_mIoU, test_FWIoU, test_loss)



        