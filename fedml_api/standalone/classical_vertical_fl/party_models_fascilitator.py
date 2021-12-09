import numpy as np
import torch
import torch.nn as nn

from fedml_api.model.finance.vfl_models_standalone import DenseModel


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class VFLGuestModel(object):

    def __init__(self, local_model):
        super(VFLGuestModel, self).__init__()
        self.localModel = local_model
        self.feature_dim = local_model.get_output_dim()
        self.is_debug = False

        self.classifier_criterion = nn.BCEWithLogitsLoss()
        self.dense_model = DenseModel(input_dim=self.feature_dim, output_dim=1, bias=True)
        self.parties_grad_component_list = []
        self.current_global_step = None
        self.X = None
        self.y = None

    def set_dense_model(self, dense_model):
        self.dense_model = dense_model

    def set_batch(self, X, y, global_step):
        self.X = X
        self.y = y
        self.current_global_step = global_step

    def _fit(self, X, y):
        self.temp_K_Z = self.localModel.forward(X)
        self.K_U = self.dense_model.forward(self.temp_K_Z)

        self._compute_common_gradient_and_loss(y)
        self._update_models(X, y)

    def predict(self, X, component_list):
        temp_K_Z = self.localModel.forward(X)
        U = self.dense_model.forward(temp_K_Z)
        for comp in component_list:
            U = U + comp
        return sigmoid(np.sum(U, axis=1))

    def receive_components(self, component_list):
        for party_component in component_list:
            self.parties_grad_component_list.append(party_component)

    def fit(self):
        self._fit(self.X, self.y)
        self.parties_grad_component_list = []

    def _compute_common_gradient_and_loss(self, y):
        U = self.K_U
        for grad_comp in self.parties_grad_component_list:
            U = U + grad_comp

        U = torch.tensor(U, requires_grad=True).float()
        y = torch.tensor(y)
        y = y.type_as(U)
        class_loss = self.classifier_criterion(U, y)
        grads = torch.autograd.grad(outputs=class_loss, inputs=U)
        self.top_grads = grads[0].numpy()
        self.loss = class_loss.item()

    def send_gradients(self):
        return self.top_grads

    def _update_models(self, X, y):
        back_grad = self.dense_model.backward(self.temp_K_Z, self.top_grads)
        self.localModel.backward(X, back_grad)

    def get_loss(self):
        return self.loss


class VFLHostModel(object):

    def __init__(self, local_model):
        super(VFLHostModel, self).__init__()
        self.localModel = local_model
        self.feature_dim = local_model.get_output_dim()
        self.is_debug = False

        self.dense_model = DenseModel(input_dim=self.feature_dim, output_dim=1, bias=False)
        self.common_grad = None
        self.partial_common_grad = None
        self.current_global_step = None
        self.X = None

    def set_dense_model(self, dense_model):
        self.dense_model = dense_model

    def set_batch(self, X, global_step):
        self.X = X
        self.current_global_step = global_step

    def _forward_computation(self, X):
        self.A_Z = self.localModel.forward(X)
        A_U = self.dense_model.forward(self.A_Z)
        return A_U

    def _fit(self, X, y):
        back_grad = self.dense_model.backward(self.A_Z, self.common_grad)
        self.localModel.backward(X, back_grad)

    def receive_gradients(self, gradients):
        self.common_grad = gradients
        self._fit(self.X, None)

    def send_components(self):
        return self._forward_computation(self.X)

    def predict(self, X):
        return self._forward_computation(X)
