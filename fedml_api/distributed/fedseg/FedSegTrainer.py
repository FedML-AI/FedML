import logging, os, time
from os import path
import torch
from torch import nn
import numpy as np
import gc
import shelve

from tqdm import tqdm

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

            if self.args.backbone_freezed:
                self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr * 10,
                                                 momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
            else:
                train_params = [{'params': self.model.get_1x_lr_params(), 'lr': args.lr},
                                {'params': self.model.get_10x_lr_params(), 'lr': args.lr * 10}]

                self.optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        else:
            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=self.args.lr, weight_decay=self.args.weight_decay, amsgrad=True)

        if self.args.backbone_freezed and self.args.extract_feat:            
            logging.info('Client:{} Generating Feature Maps for Training Dataset'.format(client_index))
            self.train_data_extracted_features_path = self._extract_features(self.train_local, 'train_features')

    def update_model(self, weights):
        
        if self.args.backbone_freezed:
            logging.info("Update_model. client_index (w\o Backbone):{}".format(self.client_index))
            self.model.encoder_decoder.load_state_dict(weights)
        else:
            logging.info("Update_model. client_index:{}".format(self.client_index))
            self.model.load_state_dict(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def _extract_features(self, dataset_loader, file_name):
        self.model.eval()
        self.model.to(self.device)

        if self.args.partition_method == "hetero":
           directory =  "./extracted_features/" + self.args.dataset + "/hetero/"
           file_path = directory + str(self.client_index) + '-' + file_name
           
        else:
            directory = "./extracted_features/" + self.args.dataset + "/homo/"
            file_path = directory + str(self.client_index) + '-' + file_name

        if not os.path.exists(directory):
            os.makedirs(directory)
        
        if not path.exists(file_path):
            logging.info('Extracting Features')
            features_db = shelve.open(file_path)
            with torch.no_grad():
                for (batch_idx, batch) in tqdm(enumerate(dataset_loader)):
                    x, labels = batch['image'], batch['label']
                    x = x.to(self.device)
                    extracted_inputs, extracted_features = self.model.feature_extractor(x)
                    features_db[str(batch_idx)] = (extracted_inputs.cpu().detach(), extracted_features.cpu().detach(), labels)
            features_db.close()
        logging.info('Client {0}: Returning extracted features database'.format(self.client_index))
        return file_path


    def train(self):

        self.model.to(self.device)

        # change to train mode
        self.model.train()
        
        # if self.args.backbone_freezed:
        logging.info('Training client {0} for {1} Epochs'.format(self.client_index, self.args.epochs))
            
        # else:
        #     logging.info('Training client (Fine-Tuning Backbone) {0} for {1} Epochs'.format(self.client_index, self.args.epochs))

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
                if (batch_idx % 100 == 0):
                    logging.info('Client Id: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.client_index, batch_idx, loss, (time.time()-t)/60))

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Client Id: {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_index,
                                                                epoch, sum(epoch_loss) / len(epoch_loss)))

            logging.info('Client Id: {0} Epoch: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.client_index, epoch, batch_loss[-1], (time.time()-t)/60))

        if self.args.backbone_freezed:
            weights = self.model.encoder_decoder.cpu().state_dict()

        else:
            weights = self.model.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)

        return weights, self.local_sample_number


    def train_extracted_feat(self):

        self.model.to(self.device)

        # change to train mode
        self.model.train()
        
        logging.info('Training client (Backbone Freezed) {0} for {1} Epochs; Feature Map already extracted'.format(self.client_index, self.args.epochs))
        epoch_loss = []
        
        for epoch in range(self.args.epochs):
            t = time.time()
            batch_loss = []

            logging.info('Client Id: {0}, Epoch: {1}'.format(self.client_index, epoch))

            with shelve.open(self.train_data_extracted_features_path, 'r') as features_db:
                for batch_idx in features_db.keys():
                    (x, low_level_feat, labels) = features_db[batch_idx]
                    x, low_level_feat, labels = x.to(self.device), low_level_feat.to(self.device), labels.to(self.device)

                    self.scheduler(self.optimizer, batch_idx, epoch)
                    self.optimizer.zero_grad()

                    log_probs = self.model.encoder_decoder(x, low_level_feat)                
                    loss = self.criterion(log_probs, labels).to(self.device)
                    loss.backward()
                    self.optimizer.step()
                    batch_loss.append(loss.item())
                    logging.info('Client Id: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.client_index, batch_idx, loss, (time.time()-t)/60))
                # if (batch_idx % 500 == 0):
                # logging.info('Client Id: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.client_index, batch_idx, loss, (time.time()-t)/60))
            features_db.close()
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info('(Client Id: {}. Local Training Epoch: {} \tLoss: {:.6f}'.format(self.client_index,
                                                                epoch, sum(epoch_loss) / len(epoch_loss)))

        weights = self.model.encoder_decoder.cpu().state_dict()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(weights)
        return weights, self.local_sample_number

    def test(self):

        self.model.eval()
        self.model.to(self.device)
        train_evaluation_metrics = None

        if self.round_idx and self.round_idx % self.args.evaluation_frequency == 0:
            if self.args.backbone_freezed and self.args.extract_feat:
                    logging.info('Evaluating client ({client_index}) model on Train dataset with extracted feature maps (Backbone Freezed)'.format(client_index = self.client_index))
                    train_evaluation_metrics = self.infer_extracted_feat(self.train_local)
            else:
                logging.info('Evaluating client ({client_index}) model on Train dataset {backbone}'.format(backbone = "(Backbone Freezed)" if self.args.backbone_freezed else "",
                                                                                                           client_index = self.client_index))
                train_evaluation_metrics = self.infer(self.train_local)

        logging.info('Testing client ({client_index}) on Test Dataset'.format(client_index = self.client_index))                
        test_evaluation_metrics = self.infer(self.test_local)
        logging.info("Testing Complete for client {}".format(self.client_index))
        self.round_idx+=1
        return train_evaluation_metrics, test_evaluation_metrics


    def infer(self, test_data):
        self.model.eval()
        self.model.to(self.device)
        t = time.time()
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
                if (batch_idx % 100 == 0):
                    logging.info('Client Id: {0} Iteration: {1}, Loss: {2}, Time Elapsed: {3}'.format(self.client_index, batch_idx, loss, (time.time()-t)/60))

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


    def infer_extracted_feat(self, test_data):
        self.model.eval()
        self.model.to(self.device)
        self.evaluator.reset()

        if test_data == self.train_local:
            test_data_extracted_features = self.train_data_extracted_features_path
        else:
            test_data_extracted_features = self.test_data_extracted_features_path

        test_acc = test_acc_class = test_mIoU = test_FWIoU = test_loss = test_total = 0.
        criterion = SegmentationLosses().build_loss(mode=self.args.loss_type)


        with torch.no_grad():
            with shelve.open(test_data_extracted_features, 'r') as features_db:
                for batch_idx in features_db.keys():
                    (x, low_level_feat, target) = features_db[batch_idx]
                    x, low_level_feat, target = x.to(self.device), low_level_feat.to(self.device), target.to(self.device)
                    output = self.model.encoder_decoder(x, low_level_feat)
                    loss = criterion(output, target).to(self.device)
                    test_loss += loss.item()
                    test_total += target.size(0)
                    pred = output.cpu().numpy()
                    target = target.cpu().numpy()
                    pred = np.argmax(pred, axis = 1)
                    self.evaluator.add_batch(target, pred)
                # logging.info("time per batch = " + str(time_end_test_per_batch - time_start_test_per_batch))
            features_db.close()
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