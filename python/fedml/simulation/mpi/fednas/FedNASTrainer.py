import logging

import torch
from torch import nn

from ....model.cv.darts import utils
from ....model.cv.darts.architect import Architect


class FedNASTrainer(object):
    """
    Federated NAS Trainer for local model training and inference.

    This class is responsible for performing local training and inference on client devices during federated NAS.

    Args:
        client_index (int): Index of the client within the federated system.
        train_data_local_dict (dict): Dictionary containing local training datasets for each client.
        test_data_local_dict (dict): Dictionary containing local test/validation datasets for each client.
        train_data_local_num (int): Number of training samples on the local client.
        train_data_num (int): Total number of training samples across all clients.
        model (nn.Module): The neural network model to be trained.
        device: The computing device (e.g., GPU) to perform training and inference.
        args: Additional configuration and hyperparameters for training and inference.

    Methods:
        update_model(weights):
            Update the model's weights with global model weights.

        update_arch(alphas):
            Update the model's architecture with global architecture parameters.

        search():
            Perform local architecture search and training.

        train():
            Perform local training.

        local_train(train_queue, valid_queue, model, criterion, optimizer):
            Perform local training on a batch of data.

        local_infer(valid_queue, model, criterion):
            Perform local inference on a batch of data.

        infer():
            Perform inference using the trained model.

    """
    def __init__(
            self,
            client_index,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num,
            train_data_num,
            model,
            device,
            args,
    ):

        self.client_index = client_index
        self.all_train_data_num = train_data_num

        self.device = device
        self.args = args
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.model = model
        self.model.to(self.device)
        self.train_local = train_data_local_dict[client_index]
        self.local_sample_number = train_data_local_num
        self.test_local = test_data_local_dict[client_index]

    def update_model(self, weights):
        """
        Update the model with new weights.

        Args:
            weights (dict): The model weights to update.
        """
        logging.info("update_model. client_index = %d" % self.client_index)
        self.model.load_state_dict(weights)

    def update_arch(self, alphas):
        """
        Update the model architecture parameters (only used in the search stage).

        Args:
            alphas (list): The architecture parameters to update.
        """
        logging.info("update_arch. client_index = %d" % self.client_index)
        for a_g, model_arch in zip(alphas, self.model.arch_parameters()):
            model_arch.data.copy_(a_g.data)

    # local search
    def search(self):
        """
        Perform local neural architecture search.

        Returns:
            tuple: A tuple containing the following elements:
                - weights (dict): The updated model weights.
                - alphas (list): The updated architecture parameters (only in the search stage).
                - local_sample_number (int): The number of local training samples.
                - local_avg_train_acc (float): The average training accuracy.
                - local_avg_train_loss (float): The average training loss.
        """
        self.model.to(self.device)
        self.model.train()

        arch_parameters = self.model.arch_parameters()
        arch_params = list(map(id, arch_parameters))

        parameters = self.model.parameters()
        weight_params = filter(lambda p: id(p) not in arch_params, parameters)

        optimizer = torch.optim.SGD(
            weight_params,  # model.parameters(),
            self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        architect = Architect(self.model, self.criterion, self.args, self.device)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min
        )

        local_avg_train_acc = []
        local_avg_train_loss = []
        for epoch in range(self.args.epochs):
            # training
            train_acc, train_obj, train_loss = self.local_search(
                self.train_local,
                self.test_local,
                self.model,
                architect,
                self.criterion,
                optimizer,
            )
            logging.info(
                "client_idx = %d, epoch = %d, local search_acc %f"
                % (self.client_index, epoch, train_acc)
            )
            local_avg_train_acc.append(train_acc)
            local_avg_train_loss.append(train_loss)

            # # validation
            # with torch.no_grad():
            #     valid_acc, valid_obj, valid_loss = self.local_infer(self.test_local, self.model, self.criterion)
            # logging.info('client_idx = %d, epoch = %d, local valid_acc %f' % (self.client_index, epoch, valid_acc))

            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info(
                "client_idx = %d, epoch %d lr %e" % (self.client_index, epoch, lr)
            )

        weights = self.model.cpu().state_dict()
        alphas = self.model.cpu().arch_parameters()

        return (
            weights,
            alphas,
            self.local_sample_number,
            sum(local_avg_train_acc) / len(local_avg_train_acc),
            sum(local_avg_train_loss) / len(local_avg_train_loss),
        )

    def local_search(
            self, train_queue, valid_queue, model, architect, criterion, optimizer
    ):
        """
        Perform local neural architecture search.

        Args:
            train_queue (DataLoader): DataLoader for the training dataset.
            valid_queue (DataLoader): DataLoader for the validation dataset.
            model (nn.Module): The neural network model.
            architect (Architect): The architect responsible for architecture search.
            criterion: The loss criterion for optimization.
            optimizer: The optimizer for weight updates.

        Returns:
            tuple: A tuple containing the following elements:
                - top1_accuracy (float): Top-1 accuracy achieved during local search.
                - loss (float): Average loss during local search.
        """
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        loss = None

        for step, (input, target) in enumerate(train_queue):

            # logging.info("epoch %d, step %d START" % (epoch, step))
            n = input.size(0)

            # model.set_tau(
            #     self.args.tau_max - self.args.epochs * 1.0 / self.args.epochs * (self.args.tau_max - self.args.tau_min))

            input = input.to(self.device)
            target = target.to(self.device)

            # get a random minibatch from the search queue with replacement
            input_search, target_search = next(iter(valid_queue))
            input_search = input_search.to(self.device)
            target_search = target_search.to(self.device)

            architect.step_v2(
                input,
                target,
                input_search,
                target_search,
                self.args.lambda_train_regularizer,
                self.args.lambda_valid_regularizer,
            )

            optimizer.zero_grad()
            logits = model(input)
            loss = criterion(logits, target)  # pylint: disable=E1102

            loss.backward()
            parameters = model.arch_parameters()
            nn.utils.clip_grad_norm_(parameters, self.args.grad_clip)
            optimizer.step()

            # logging.info("step %d. update weight by SGD. FINISH\n" % step)
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # torch.cuda.empty_cache()

            if step % self.args.report_freq == 0:
                logging.info(
                    "client_index = %d, search %03d %e %f %f",
                    self.client_index,
                    step,
                    objs.avg,
                    top1.avg,
                    top5.avg,
                )

        return top1.avg / 100.0, objs.avg / 100.0, loss

    def train(self):
        """
        Perform local training.

        Returns:
            tuple: A tuple containing the following elements:
                - weights (dict): The updated model weights.
                - local_sample_number (int): The number of local training samples.
                - local_avg_train_acc (float): The average training accuracy.
                - local_avg_train_loss (float): The average training loss.
        """
        self.model.to(self.device)
        self.model.train()

        parameters = self.model.parameters()

        optimizer = torch.optim.SGD(
            parameters,  # model.parameters(),
            self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, float(self.args.epochs), eta_min=self.args.learning_rate_min
        )

        local_avg_train_acc = []
        local_avg_train_loss = []
        for epoch in range(self.args.epochs):
            # training
            train_acc, train_obj, train_loss = self.local_train(
                self.train_local, self.test_local, self.model, self.criterion, optimizer
            )
            logging.info(
                "client_idx = %d, local train_acc %f" % (self.client_index, train_acc)
            )
            local_avg_train_acc.append(train_acc)
            local_avg_train_loss.append(train_loss)

            scheduler.step()
            lr = scheduler.get_lr()[0]
            logging.info(
                "client_idx = %d, epoch %d lr %e" % (self.client_index, epoch, lr)
            )

        weights = self.model.cpu().state_dict()

        return (
            weights,
            self.local_sample_number,
            sum(local_avg_train_acc) / len(local_avg_train_acc),
            sum(local_avg_train_loss) / len(local_avg_train_loss),
        )

    def local_train(self, train_queue, valid_queue, model, criterion, optimizer):
        """
        Perform local training.

        Args:
            train_queue (DataLoader): DataLoader for the training dataset.
            valid_queue (DataLoader): DataLoader for the validation dataset.
            model (nn.Module): The neural network model.
            criterion: The loss criterion for optimization.
            optimizer: The optimizer for weight updates.

        Returns:
            tuple: A tuple containing the following elements:
                - top1_accuracy (float): Top-1 accuracy achieved during local training.
                - loss (float): Average loss during local training.
        """
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()

        for step, (input, target) in enumerate(train_queue):
            # logging.info("epoch %d, step %d START" % (epoch, step))
            model.train()
            n = input.size(0)

            input = input.to(self.device)
            target = target.to(self.device)

            optimizer.zero_grad()
            logits, logits_aux = model(input)
            loss = criterion(logits, target)  # pylint: disable=E1102
            if self.args.auxiliary:
                loss_aux = criterion(logits_aux, target)  # pylint: disable=E1102
                loss += self.args.auxiliary_weight * loss_aux
            loss.backward()
            parameters = model.parameters()
            nn.utils.clip_grad_norm_(parameters, self.args.grad_clip)
            optimizer.step()
            # logging.info("step %d. update weight by SGD. FINISH\n" % step)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # torch.cuda.empty_cache()
            if step % self.args.report_freq == 0:
                logging.info("train %03d %e %f %f", step, objs.avg, top1.avg, top5.avg)

        return top1.avg, objs.avg, loss

    def local_infer(self, valid_queue, model, criterion):
        """
        Perform local inference.

        Args:
            valid_queue (DataLoader): DataLoader for the validation dataset.
            model (nn.Module): The neural network model.
            criterion: The loss criterion for evaluation.

        Returns:
            tuple: A tuple containing the following elements:
                - top1_accuracy (float): Top-1 accuracy achieved during local inference.
                - loss (float): Average loss during local inference.
        """
        objs = utils.AvgrageMeter()
        top1 = utils.AvgrageMeter()
        top5 = utils.AvgrageMeter()
        model.eval()
        loss = None
        for step, (input, target) in enumerate(valid_queue):
            input = input.to(self.device)
            target = target.to(self.device)

            logits = model(input)
            loss = criterion(logits, target)  # pylint: disable=E1102

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.args.report_freq == 0:
                logging.info(
                    "client_index = %d, valid %03d %e %f %f",
                    self.client_index,
                    step,
                    objs.avg,
                    top1.avg,
                    top5.avg,
                )

        return top1.avg / 100.0, objs.avg / 100.0, loss

    # after searching, infer() function is used to infer the searched architecture
    def infer(self):
        """
        Perform inference using the trained model.

        Returns:
            tuple: A tuple containing the following elements:
                - test_accuracy (float): Test accuracy achieved using the trained model.
                - test_loss (float): Test loss using the trained model.
        """
        self.model.to(self.device)
        self.model.eval()

        test_correct = 0.0
        test_loss = 0.0
        test_sample_number = 0.0
        test_data = self.train_local
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(self.device)
                target = target.to(self.device)

                pred = self.model(x)
                loss = self.criterion(pred, target)  # pylint: disable=E1102
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                test_correct += correct.item()
                test_loss += loss.item() * target.size(0)
                test_sample_number += target.size(0)
            logging.info(
                "client_idx = %d, local_train_loss = %s"
                % (self.client_index, test_loss)
            )
        return test_correct / test_sample_number, test_loss
