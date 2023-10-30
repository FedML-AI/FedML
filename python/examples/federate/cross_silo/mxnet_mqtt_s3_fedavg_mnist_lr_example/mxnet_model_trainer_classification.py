import logging

from mxnet import npx
from fedml.core import ClientTrainer
import mxnet as mx
from mxnet import gluon, autograd
from mxnet import nd


class MxModelTrainerCLS(ClientTrainer):
    static_model = None
    static_optimizer = None
    npx.set_np()

    def __init__(self, model, args):
        super().__init__(model, args)
        self.mx_trainer = None
        self.mx_metric = None
        self.mx_loss = None
        self.init_trainer()

    def get_model_params(self):
        return self.model.get_params()

    def set_model_params(self, model_parameters):
        if model_parameters is None:
            return

        self.model.set_params(model_parameters)

        self.reset_trainer()

    def init_trainer(self):
        init_x = nd.random.uniform(shape=(self.model.output_dim, self.model.input_dim))
        init_x = init_x.as_np_ndarray()
        self.model(init_x)

    def reset_trainer(self):
        self.mx_trainer = gluon.Trainer(self.model.get_layer_params(), self.args.client_optimizer,
                                        {'learning_rate': self.args.learning_rate})
        self.mx_metric = mx.gluon.metric.Accuracy()
        self.mx_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    def train_method2(self, train_data, device, args):
        trainer = gluon.Trainer(self.model.get_layer_params(), args.client_optimizer, {"learning_rate": args.learning_rate})
        accuracy_metric = mx.gluon.metric.Accuracy()
        loss_metric = mx.gluon.metric.CrossEntropy()
        metrics = mx.gluon.metric.CompositeEvalMetric()
        for child_metric in [accuracy_metric, loss_metric]:
            metrics.add(child_metric)
        loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
        for epoch in range(args.epochs):
            num_examples = 0
            for batch_idx, (x, labels) in enumerate(train_data):
                data = gluon.utils.split_and_load(
                    x, ctx_list=[device], batch_axis=0
                )
                label = gluon.utils.split_and_load(
                    labels, ctx_list=[device], batch_axis=0
                )
                outputs = []
                with autograd.record():
                    for x_item, y_item in zip(data, label):
                        z = self.model(x_item)
                        loss = loss_func(z, y_item)
                        loss.backward()
                        outputs.append(npx.softmax(z))
                        num_examples += len(x_item)
                metrics.update(label, outputs)
                trainer.step(x.shape[0])
            trainings_metric = metrics.get_name_value()
            acc_val, loss_val = trainings_metric

            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    self.id, epoch, loss_val[1], acc_val[1]
                )
            )

    def train(self, train_data, device, args):
        self.model.reset_device(device)

        epoch_acc = []
        epoch_loss = []

        for epoch in range(args.epochs):
            total_loss = 0.0
            self.mx_metric.reset()
            for batch_idx, (x, labels) in enumerate(train_data):
                # Do SGD on a batch of training samples.
                x = x.to_device(device)
                labels = labels.to_device(device)

                with autograd.record():
                    output = self.model(x)
                    loss_func = self.mx_loss(output, labels)
                    loss_func.backward()
                    total_loss += loss_func.mean()

                self.mx_trainer.step(x.shape[0])
                self.mx_metric.update([labels], [output])

            _, accuracy = self.mx_metric.get()
            loss = total_loss/len(train_data)
            epoch_acc.append(accuracy)
            epoch_loss.append(loss)

            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
                )
            )
