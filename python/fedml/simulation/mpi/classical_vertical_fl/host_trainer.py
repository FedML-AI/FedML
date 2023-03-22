import torch
from torch import nn, optim


class HostTrainer(object):
    def __init__(
        self,
        client_index,
        device,
        X_train,
        X_test,
        model_feature_extractor,
        model_classifier,
        args,
    ):
        # device information
        self.client_index = client_index
        self.device = device
        self.args = args

        # training dataset
        self.X_train = X_train
        self.X_test = X_test
        self.batch_size = args.batch_size

        N = self.X_train.shape[0]
        residual = N % args.batch_size
        if residual == 0:
            self.n_batches = N // args.batch_size
        else:
            self.n_batches = N // args.batch_size + 1
        # logging.info("n_batches = %d" % self.n_batches)
        self.batch_idx = 0

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

        self.cached_extracted_features = None

    def get_batch_num(self):
        return self.n_batches

    def computer_logits(self, round_idx):
        batch_x = self.X_train[
            self.batch_idx * self.batch_size : self.batch_idx * self.batch_size
            + self.batch_size
        ]
        self.batch_x = torch.tensor(batch_x).float().to(self.device)
        self.extracted_feature = self.model_feature_extractor.forward(self.batch_x)
        logits = self.model_classifier.forward(self.extracted_feature)
        # copy to CPU host memory
        logits_train = logits.cpu().detach().numpy()
        self.batch_idx += 1
        if self.batch_idx == self.n_batches:
            self.batch_idx = 0

        # for test
        if (round_idx + 1) % self.args.frequency_of_the_test == 0:
            X_test = torch.tensor(self.X_test).float().to(self.device)
            extracted_feature = self.model_feature_extractor.forward(X_test)
            logits_test = self.model_classifier.forward(extracted_feature)
            logits_test = logits_test.cpu().detach().numpy()
        else:
            logits_test = None

        return logits_train, logits_test

    def update_model(self, gradient):
        # logging.info("#######################gradient = " + str(gradient))
        gradient = torch.tensor(gradient).float().to(self.device)
        back_grad = self._bp_classifier(self.extracted_feature, gradient)
        self._bp_feature_extractor(self.batch_x, back_grad)

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
