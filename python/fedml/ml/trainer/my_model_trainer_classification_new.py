import torch
from torch import nn

from ...core.alg_frame.client_trainer import ClientTrainer
import logging

from .client_optimizer_creator import create_client_optimizer
from .local_cache import FedMLLocalCache

from fedml.ml.ml_message import MLMessage
from fedml.utils.model_utils import transform_tensor_to_list, transform_list_to_tensor



class ModelTrainerCLS(ClientTrainer):

    def __init__(self, model, args):
        super().__init__(model, args)
        self.client_status = {}

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def set_client_index(self, client_index):
        self.client_index = client_index

    def set_server_result(self, server_result):
        weights = server_result[MLMessage.MODEL_PARAMS]
        if self.args.is_mobile == 1:
            weights = transform_list_to_tensor(weights)

        self.set_model_params(weights)
        # self.params_to_client_optimizer = server_result[MLMessage.PARAMS_TO_CLIENT_OPTIMIZER]
        self.server_result = server_result

    def load_client_status(self):
        if self.args.local_cache:
            if self.args.client_cache == "stateful":
                """
                In this mode, ClientTrainer is stateful during the whole training process.
                """
                client_status = self.client_status
            elif self.args.client_cache == "localhost":
                client_status = FedMLLocalCache.load(self.args, self.client_index)
            else:
                raise NotImplementedError
        else:
            client_status = {}
        return client_status


    def save_client_status(self, client_status={}):
        if self.args.local_cache:
            if client_status is None or len(client_status) == 0:
                return
            if self.args.client_cache == "stateful":
                """
                In this mode, ClientTrainer is stateful during the whole training process.
                """
                self.client_status = client_status
            elif self.args.client_cache == "localhost":
                FedMLLocalCache.save(self.args, self.client_index, client_status)
            else:
                raise NotImplementedError
        else:
            pass


    def train(self, train_data, device, args):
        model = self.model

        client_optimizer = create_client_optimizer(args)

        model.to(device)
        model.train()
        # client_optimizer.

        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102

        client_status = self.load_client_status()
        client_optimizer.load_status(args, client_status)
        client_optimizer.preprocess(args, self.client_index, model, train_data, device, self.server_result, criterion)


        # train and update
        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss = client_optimizer.backward(args, self.client_index, model, x, labels, criterion, device, loss)
                client_optimizer.update(args, self.client_index, model, x, labels, criterion, device)
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )
        client_result = {}
        # weights_or_grads, params_to_server_optimizer = client_optimizer.end_local_training(args, self.client_index, model, train_data, device)
        other_result = client_optimizer.end_local_training(args, self.client_index, model, train_data, device)
        client_result.update(other_result)
        # transform Tensor to list
        if self.args.is_mobile == 1:
            client_result[MLMessage.MODEL_PARAMS] = transform_tensor_to_list(client_result[MLMessage.MODEL_PARAMS])

        new_client_status = {"default": 0}
        new_client_status = client_optimizer.add_status(new_client_status)
        self.save_client_status(new_client_status)

        return sum(epoch_loss) / len(epoch_loss), client_result


    def train_iterations(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []

        current_steps = 0
        current_epoch = 0
        while current_steps < args.local_iterations:
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info(
                #     "Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                #         epoch,
                #         (batch_idx + 1) * args.batch_size,
                #         len(train_data) * args.batch_size,
                #         100.0 * (batch_idx + 1) / len(train_data),
                #         loss.item(),
                #     )
                # )
                batch_loss.append(loss.item())
                current_steps += 1
                if current_steps == args.local_iterations:
                    break
            current_epoch += 1
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, current_epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )


    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
        return metrics
