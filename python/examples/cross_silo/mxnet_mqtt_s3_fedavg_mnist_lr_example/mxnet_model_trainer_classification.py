import logging

from fedml.core import ClientTrainer
import mxnet as mx
from mxnet import gluon, autograd


class MxModelTrainerCLS(ClientTrainer):
    static_model = None
    static_optimizer = None

    def __init__(self, model, args):
        super().__init__(model, args)
        self.optimizer = None
        self.mx_trainer = None
        self.mx_metric = None
        self.mx_loss = None
        self.init_trainer()

    def get_model_params(self):
        return self.model.collect_params()

    def set_model_params(self, model_parameters):
        if self.optimizer is None:
            return

        self.model.share_parameters(model_parameters)

    def init_trainer(self):
        # Trainer is for updating parameters with gradient.
        self.mx_trainer = gluon.Trainer(self.model.collect_params(), self.args.client_optimizer,
                                        {'learning_rate': self.args.learning_rate})
        self.mx_metric = mx.gluon.metric.Accuracy()
        self.mx_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    def train(self, train_data, device, args):
        self.model.reset_device(device)

        epoch_acc = []
        epoch_loss = []

        for epoch in range(args.epochs):
            total_loss = 0.0
            self.mx_metric.reset()
            for batch_idx, (x, labels) in enumerate(train_data):
                # Do SGD on a batch of training samples.
                with device:
                    x = x.numpy()
                    labels = labels.numpy()

                with autograd.record():
                    output = self.model(x)
                    loss_func = self.mx_loss(output, labels)
                    loss_func.backward()
                    total_loss += loss_func.mean()

                self.mx_trainer.step(x.shape[0])
                self.mx_metric.update([labels], [output])

            accuracy = self.mx_metric.get()
            loss = total_loss/len(train_data)
            epoch_acc.append(accuracy)
            epoch_loss.append(loss)

            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
                )
            )
