import logging, os, time
from os import path
import torch
from torch import nn
import numpy as np

from fedml_api.distributed.fedseg.utils import transform_tensor_to_list, SegmentationLosses, Evaluator, LR_Scheduler, EvaluationMetricsKeeper, save_as_pickle_file, load_from_pickle_file

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
        self.round_idx = 0
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

        if self.args.backbone_freezed:
            self.train_data_extracted_features, self.test_data_extracted_features = self._extract_features()


    def update_model(self, weights):
        
        if self.args.backbone_freezed:
            logging.info("update_model. client_index (w\o Backbone) = %d" % self.client_index)
            self.model.head.load_state_dict(weights)
        else:
            logging.info("update_model. client_index = %d" % self.client_index)
            self.model.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def _extract_features(self):
        self.model.eval()
        self.model.to(self.device)

        if self.args.partition_method == "hetero":
           directory_train =  "./extracted_features/" + self.args.dataset + "/hetero/"

           path_train = directory_train + str(self.client_index) + "-train.pkl"
           
           directory_test = "./extracted_features/" + self.args.dataset + "/hetero/"
           path_test = directory_test + str(self.client_index) + "-test.pkl"

        else:
            directory_train = "./extracted_features/" + self.args.dataset + "/homo/"
            path_train = directory_train + str(self.client_index) + "-train.pkl"
            
            directory_test = "./extracted_features/" + self.args.dataset + "/homo/"
            path_test = directory_test + str(self.client_index) + "-test.pkl"

        if not os.path.exists(directory_train):
            os.makedirs(directory_train)

        if not os.path.exists(directory_test):
            os.makedirs(directory_test)
        
        train_data_extracted_features = dict()
        test_data_extracted_features = dict()

        if path.exists(path_train):
            logging.info('Loading Extracted Features for Training Dataset')
            train_data_extracted_features = load_from_pickle_file(path_train)

        else:
            logging.info('Extracting Features for Training Dataset')
            with torch.no_grad():
                for (batch_idx, batch) in enumerate(self.train_local):
                    time_start_train_per_batch = time.time()
                    x, labels = batch['image'], batch['label']
                    x = x.to(self.device)
                    extracted_inputs, extracted_features = self.model.transformer(x)
                    # DEBUG: Verify dimensions
                    train_data_extracted_features[batch_idx] = (extracted_inputs.cpu().detach(), extracted_features.cpu().detach(), labels)
                    time_end_train_per_batch = time.time()

                    # logging.info("train_local feature extraction - client_id={}, batch_id={}, time per batch = {}".format(self.client_index, batch_idx, str(
                    #     time_end_train_per_batch - time_start_train_per_batch)))

                save_as_pickle_file(path_train, train_data_extracted_features)


        if path.exists(path_test):
            logging.info('Loading Extracted Features for Testing Dataset')
            test_data_extracted_features = load_from_pickle_file(path_test)

        else:
            logging.info('Extracting Features for Testing Dataset')
            with torch.no_grad():
                for (batch_idx, batch) in enumerate(self.test_local):
                    time_start_test_per_batch = time.time()
                    x, labels = batch['image'], batch['label']
                    x = x.to(self.device)
                    extracted_inputs, extracted_features = self.model.transformer(x)
                    # DEBUG: Verify dimensions
                    test_data_extracted_features[batch_idx] = (extracted_inputs.cpu().detach(), extracted_features.cpu().detach(), labels)
                    time_end_test_per_batch = time.time()

                    # logging.info("test_local feature extraction - time per batch = " + str(
                    #     time_end_test_per_batch - time_start_test_per_batch))

                save_as_pickle_file(path_test, test_data_extracted_features)

        return train_data_extracted_features, test_data_extracted_features

    def train(self):

        self.model.to(self.device)
        # change to train mode
        self.model.train()
        
        logging.info('Training client (w/o Backbone) {0} for {1} Epochs'.format(self.client_index, self.args.epochs))
        epoch_loss = []
        
        for epoch in range(self.args.epochs):
            t = time.time()
            batch_loss = []

            logging.info('Client Id: {0}, Epoch: {1}'.format(self.client_index, epoch))

            for batch_idx in self.train_data_extracted_features.keys():

                (x, low_level_feat, labels) = self.train_data_extracted_features[batch_idx]
                x, low_level_feat, labels = x.to(self.device), low_level_feat.to(self.device), labels.to(self.device)

                self.scheduler(self.optimizer, batch_idx, epoch)
                self.optimizer.zero_grad()

                log_probs = self.model.head(x, low_level_feat)                
                loss = self.criterion(log_probs, labels).to(self.device)
                loss.backward()
                self.optimizer.step()
                batch_loss.append(loss.item())

                # if (batch_idx % 500 == 0):
                # logging.info('Client Id: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.client_index, batch_idx, loss, (time.time()-t)/60))

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Client Id: {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_index,
                                                                epoch, sum(epoch_loss) / len(epoch_loss)))

            logging.info('Client Id: {0} Epoch: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.client_index, epoch, batch_loss[-1], (time.time()-t)/60))

        weights = self.model.head.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        self.round_idx+=1
        return weights, self.local_sample_number

    def _train_raw_data(self):

        self.model.to(self.device)
        # change to train mode
        self.model.train()
        
        logging.info('Training client {0} for {1} Epochs'.format(self.client_index, self.args.epochs))
        epoch_loss = []

        # for (batch_ix, batch) in enumerate(self.train_local):
        #     logging.info('Train Batch ID: {}, Images: {}, Masks: {}'.format(batch_ix, batch['image'].shape, batch['label'].shape))

        # for (batch_ix, batch) in enumerate(self.test_local):
        #     logging.info('Test Batch ID: {}, Images: {}, Masks: {}'.format(batch_ix, batch['image'].shape, batch['label'].shape))

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
                # if (batch_idx % 500 == 0):
                logging.info('Client Id: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.client_index, batch_idx, loss, (time.time()-t)/60))
                if batch_idx == 10:
                    break

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Client Id: {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_index,
                                                                epoch, sum(epoch_loss) / len(epoch_loss)))

            logging.info('Client Id: {0} Epoch: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.client_index, epoch, batch_loss[-1], (time.time()-t)/60))

        weights = self.model.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        self.round_idx+=1
        return weights, self.local_sample_number
    

    def test(self):

        self.model.eval()
        self.model.to(self.device)
        train_evaluation_metrics = None

        # Train Data
        if self.args.backbone_freezed:
            logging.info('Testing client (w/o Backbone) {0}'.format(self.client_index))
            if self.round_idx % self.args.frequency_of_the_test == 0:
                train_evaluation_metrics = self._infer(self.train_local)
            test_evaluation_metrics = self._infer(self.test_local)

        else:
            logging.info('Testing client {0} on train dataset'.format(self.client_index))
            if self.round_idx % self.args.frequency_of_the_test == 0:
                train_evaluation_metrics = self._infer_on_raw_data(self.train_local)
            test_evaluation_metrics = self._infer_on_raw_data(self.test_local)

        # Test Data        
        logging.info("Testing Complete for client {}".format(self.client_index))
        # Test on training dataset
        return train_evaluation_metrics, test_evaluation_metrics


    def _infer(self, test_data):
        time_start_test_per_batch = time.time()
        self.model.eval()
        self.model.to(self.device)
        self.evaluator.reset()

        if test_data == self.train_local:
            test_data_extracted_features = self.train_data_extracted_features
        else:
            test_data_extracted_features = self.test_data_extracted_features

        test_acc = test_acc_class = test_mIoU = test_FWIoU = test_loss = test_total = 0.
        criterion = SegmentationLosses().build_loss(mode=self.args.loss_type)


        with torch.no_grad():
            for batch_idx in test_data_extracted_features.keys():
                (x, low_level_feat, target) = self.train_data_extracted_features[batch_idx]
                x, low_level_feat, target = x.to(self.device), low_level_feat.to(self.device), target.to(self.device)
                output = self.model.head(x, low_level_feat)
                loss = criterion(output, target).to(self.device)
                test_loss += loss.item()
                test_total += target.size(0)
                pred = output.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis = 1)
                self.evaluator.add_batch(target, pred)
                time_end_test_per_batch = time.time()
                # logging.info("time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))

        # Evaluation Metrics (Averaged over number of samples)
        test_acc = self.evaluator.Pixel_Accuracy()
        test_acc_class = self.evaluator.Pixel_Accuracy_Class()
        test_mIoU = self.evaluator.Mean_Intersection_over_Union()
        test_FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        test_loss = test_loss / test_total
        
        return EvaluationMetricsKeeper(test_acc, test_acc_class, test_mIoU, test_FWIoU, test_loss)


    def _infer_on_raw_data(self, test_data):
        time_start_test_per_batch = time.time()
        self.evaluator.reset()        
        test_acc = test_acc_class = test_mIoU = test_FWIoU = test_loss = test_total = 0.
        criterion = SegmentationLosses().build_loss(mode=self.args.loss_type)

        with torch.no_grad():
            for (batch_idx, batch) in enumerate(test_data):
                x, target = batch['image'], batch['label']
                x, target = x.to(self.device), target.to(self.device)
                output = self.model(x)
                loss = criterion(output, target).to(self.device)
                test_loss += loss.item()
                test_total += target.size(0)
                pred = output.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis = 1)
                self.evaluator.add_batch(target, pred)
                if batch_idx == 10:
                    break
                # time_end_test_per_batch = time.time()
                # logging.info("time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))
                # logging.info("Client = {0} Batch = {1}".format(self.client_index, batch_idx)
                                                                            
        # Evaluation Metrics (Averaged over number of samples)
        test_acc = self.evaluator.Pixel_Accuracy()
        test_acc_class = self.evaluator.Pixel_Accuracy_Class()
        test_mIoU = self.evaluator.Mean_Intersection_over_Union()
        test_FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        test_loss = test_loss / test_total

        logging.info("Client={0}, test_acc={1}, test_acc_class={2}, test_mIoU={3}, test_FWIoU={4}, test_loss={5}".format(
            self.client_index, test_acc, test_acc_class, test_mIoU, test_FWIoU, test_loss))
        
        eval_metrics = EvaluationMetricsKeeper(test_acc, test_acc_class, test_mIoU, test_FWIoU, test_loss)
        return eval_metrics
