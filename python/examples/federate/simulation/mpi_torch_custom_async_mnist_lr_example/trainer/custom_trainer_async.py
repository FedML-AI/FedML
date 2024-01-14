import logging
import time
import torch

import numpy as np

from scipy.stats import expon
from torch import nn

from fedml.core.alg_frame.client_trainer import ClientTrainer

class CustomTrainerAsync(ClientTrainer):

    def __init__(self, model, args=None):
        super().__init__(model, args)

        # In classical FL, the time a client requires to perform local 
        # training is proportional to the client's number of examples.
        # For our simulation here, since clients have an equal number 
        # of samples, we assume the time delay is proportional to the 
        # client's rank in the FL environment.

        # Formally, let (ni)/(ri) be the (number of examples)/(rank) 
        # of client i, and let Ti be the amount of time required by 
        # client i to perform local training, with λ > 0 being a
        # constant number controlling stragglers' behavior:
        #   T(n) ~ exp(1/λn) OR 
        #   T(r) ~ exp(1/λr)

        # Smulation References:
        # - Speeding Up Distributed Machine Learning Using Codes 
        #   https://arxiv.org/pdf/1512.02673.pdf
        # - Federated Learning with Buffered Asynchronous Aggregation
        #   https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf 
        self.expon_lambda = 0.5
        self.expon_scale = np.divide(1, self.expon_lambda)
        self.sleep_time_max = args.sleep_time_max

    def get_model_params(self):
        return self.model.state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        # Adding following line to emulate random sleeping patterns 
        # across clients for testing asynchronous FL strategies.
        # x is client's rank
        sleep_time = expon.pdf(args.rank + 1, self.expon_scale) 
        sleep_time *= self.sleep_time_max
        sleep_time = np.round(sleep_time, 2)
        logging.info("Client: {}, Sleep time: {} secs".format(args.rank, sleep_time))
        time.sleep(sleep_time)
        model = self.model

        model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                # logging.info(images.shape)
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                if batch_idx % 100 == 0:
                    logging.info(
                        "Epoch: {}/{} | Batch: {}/{} | Loss: {}".format(
                            epoch + 1,
                            args.epochs,
                            batch_idx,
                            len(train_data),
                            loss.item(),
                        )
                    )

            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    "(Trainer_ID {}. Local Training Epoch: {} \tLoss: {:.6f}".format(
                        self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                    )
                )
