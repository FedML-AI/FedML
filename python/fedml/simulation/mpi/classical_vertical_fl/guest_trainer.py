import logging

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, accuracy_score
from torch import optim, nn


class GuestTrainer(object):
    def __init__(
        self,
        client_num,
        device,
        X_train,
        y_train,
        X_test,
        y_test,
        model_feature_extractor,
        model_classifier,
        args,
    ):
        self.client_num = client_num
        self.args = args
        self.device = device

        # training dataset
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = args.batch_size

        N = self.X_train.shape[0]
        residual = N % args.batch_size
        if residual == 0:
            self.n_batches = N // args.batch_size
        else:
            self.n_batches = N // args.batch_size + 1
        self.batch_idx = 0
        logging.info("number of sample = %d" % N)
        logging.info("batch_size = %d" % self.batch_size)
        logging.info("number of batches = %d" % self.n_batches)

        # model
        self.model_feature_extractor = model_feature_extractor
        self.model_feature_extractor.to(device)
        self.optimizer_fe = optim.SGD(
            self.model_feature_extractor.parameters(),
            momentum=0.9,
            weight_decay=0.01,
            lr=self.args.lr,
        )

        self.model_classifier = model_classifier
        self.model_classifier.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer_classifier = optim.SGD(
            self.model_classifier.parameters(),
            momentum=0.9,
            weight_decay=0.01,
            lr=self.args.lr,
        )

        self.host_local_train_logits_list = dict()
        self.host_local_test_logits_list = dict()

        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.loss_list = list()

    def get_batch_num(self):
        return self.n_batches

    def add_client_local_result(self, index, host_train_logits, host_test_logits):
        # logging.info("add_client_local_result. index = %d" % index)
        self.host_local_train_logits_list[index] = host_train_logits
        self.host_local_test_logits_list[index] = host_test_logits
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def train(self, round_idx):
        batch_x = self.X_train[
            self.batch_idx * self.batch_size : self.batch_idx * self.batch_size
            + self.batch_size
        ]
        batch_y = self.y_train[
            self.batch_idx * self.batch_size : self.batch_idx * self.batch_size
            + self.batch_size
        ]
        batch_x = torch.tensor(batch_x).float().to(self.device)
        batch_y = torch.tensor(batch_y).float().to(self.device)

        extracted_feature = self.model_feature_extractor.forward(batch_x)
        guest_logits = self.model_classifier.forward(extracted_feature)
        self.batch_idx += 1
        if self.batch_idx == self.n_batches:
            self.batch_idx = 0

        guest_logits = guest_logits.cpu().detach().numpy()
        for k in self.host_local_train_logits_list.keys():
            host_logits = self.host_local_train_logits_list[k]
            guest_logits += host_logits

        guest_logits = (
            torch.tensor(guest_logits, requires_grad=True).float().to(self.device)
        )
        batch_y = batch_y.type_as(guest_logits)

        # calculate the gradient until the logits for hosts
        class_loss = self.criterion(guest_logits, batch_y)  # pylint: disable=E1102
        grads = torch.autograd.grad(outputs=class_loss, inputs=guest_logits)

        loss = class_loss.item()
        self.loss_list.append(loss)

        # continue BP
        back_grad = self._bp_classifier(extracted_feature, grads)
        self._bp_feature_extractor(batch_x, back_grad)

        gradients_to_hosts = grads[0].cpu().detach().numpy()
        # logging.info("gradients_to_hosts = " + str(gradients_to_hosts))

        # for test
        if (round_idx + 1) % self.args.frequency_of_the_test == 0:
            self._test(round_idx)

        return gradients_to_hosts

    def _bp_classifier(self, x, grads):
        x = x.clone().detach().requires_grad_(True)
        output = self.model_classifier(x)
        output.backward(gradient=grads)
        x_grad = x.grad
        self.optimizer_classifier.step()
        self.optimizer_classifier.zero_grad()
        return x_grad

    def _bp_feature_extractor(self, x, grads):
        output = self.model_feature_extractor(x)
        output.backward(gradient=grads)
        self.optimizer_fe.step()
        self.optimizer_fe.zero_grad()

    def _test(self, round_idx):
        X_test = torch.tensor(self.X_test).float().to(self.device)
        y_test = self.y_test

        extracted_feature = self.model_feature_extractor.forward(X_test)
        guest_logits = self.model_classifier.forward(extracted_feature)

        guest_logits = guest_logits.cpu().detach().numpy()
        for k in self.host_local_test_logits_list.keys():
            host_logits = self.host_local_test_logits_list[k]
            guest_logits += host_logits
        y_prob_preds = self._sigmoid(np.sum(guest_logits, axis=1))

        threshold = 0.5
        y_hat_lbls, statistics = self._compute_correct_prediction(
            y_targets=y_test, y_prob_preds=y_prob_preds, threshold=threshold
        )
        acc = accuracy_score(y_test, y_hat_lbls)
        auc = roc_auc_score(y_test, y_prob_preds)
        ave_loss = np.mean(self.loss_list)
        self.loss_list = list()
        logging.info(
            "--- round_idx: {%d}, loss: {%s}, acc: {%s}, auc: {%s}"
            % (round_idx, str(ave_loss), str(acc), str(auc))
        )
        logging.info(
            precision_recall_fscore_support(
                y_test, y_hat_lbls, average="macro", warn_for=tuple()
            )
        )

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _compute_correct_prediction(self, y_targets, y_prob_preds, threshold=0.5):
        y_hat_lbls = []
        pred_pos_count = 0
        pred_neg_count = 0
        correct_count = 0
        for y_prob, y_t in zip(y_prob_preds, y_targets):
            if y_prob <= threshold:
                pred_neg_count += 1
                y_hat_lbl = 0
            else:
                pred_pos_count += 1
                y_hat_lbl = 1
            y_hat_lbls.append(y_hat_lbl)
            if y_hat_lbl == y_t:
                correct_count += 1

        return np.array(y_hat_lbls), [pred_pos_count, pred_neg_count, correct_count]
