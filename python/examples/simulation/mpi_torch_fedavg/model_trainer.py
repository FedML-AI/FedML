import torch
from torch import nn

from fedml.core.alg_frame.client_trainer import ClientTrainer
import logging


class ModelTrainerCLS(ClientTrainer):

    def __init__(self, model, args):
        super().__init__(model, args)
        self.client_status = {}

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)


    def train(self, train_data, device, args, **kwargs):
        # model = self.model

        client_optimizer = kwargs["client_optimizer"]

        self.model.to(device)
        self.model.train()
        # client_optimizer.
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
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102
        client_optimizer.preprocess(args, self.client_index,
                                    self.model, train_data,
                                    device, optimizer, criterion)

        # train and update
        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # if batch_idx > 2:
                #     break

                x, labels = x.to(device), labels.to(device)
                self.model.zero_grad()
                log_probs = self.model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss = client_optimizer.backward(args, self.client_index, self.model, x, labels, criterion, device, loss)
                client_optimizer.update(args, self.client_index, self.model, x, labels, criterion, device)
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
        local_loss = sum(epoch_loss) / len(epoch_loss)


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



















