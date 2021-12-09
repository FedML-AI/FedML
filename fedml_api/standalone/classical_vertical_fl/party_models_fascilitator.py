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
        self.is_debug = True
        self.classifier_criterion = nn.BCEWithLogitsLoss()
        self.current_global_step = None
        self.y = None

    # Left this in here in case we want to add different layers  after receiving the concatination
    def _fit(self, inter_act):
        self.final_act = self.localModel.forward(inter_act)
        return self.final_act

    def receive_concatination(self, concat_act):
        if self.localModel == None:
            if self.is_debug: print("Step 5:receives the activation")
            self.final_act = concat_act
        else:
            if self.is_debug: print("Step 5:receives the activation, preforms forward ")
            self.final_act = self._fit(concat_act)
        self._compute_loss(self.y)

    def set_batch(self, y, global_step):
        self.y = y
        self.current_global_step = global_step

    def predict(self,final_act):
        if self.localModel == None:
            if self.is_debug: print("Step 5:receives the activation")
            self.final_act = final_act
        else:
            if self.is_debug: print("Step 5:receives the activation, preforms forward ")
            self.final_act = self._fit(final_act)
        return sigmoid(np.sum(final_act, axis=1))

    def _compute_loss(self, y):
        if self.is_debug: print("Step 6:computes loss ")
        U = torch.tensor(self.final_act, requires_grad=True).float()
        y = torch.tensor(y)
        y = y.type_as(U)
        class_loss = self.classifier_criterion(U, y)
        grads = torch.autograd.grad(outputs=class_loss, inputs=U)
        self.top_grads = grads[0].numpy()
        self.loss = class_loss.item()

    def _fit_back(self, X,back_grad):
        self.localModel.backward(X, back_grad)

    def receive_gradients(self,backprop):
        self._fit_back(self.X, backprop)

    def send_loss(self):
        if self.is_debug: print("Step 7:sends loss")
        return self.loss

    def send_gradients(self):
        if self.is_debug: print("Step 8:send gradients")
        return self.top_grads

class VFLHostModel(object):

    def __init__(self, local_model):
        super(VFLHostModel, self).__init__()
        self.localModel = local_model
        self.is_debug = True
        self.common_grad = None
        self.current_global_step = None
        self.X = None

    def set_batch(self, X, global_step):
        if self.is_debug: print("Step 1: Set batch for Host")
        self.X = X
        self.current_global_step = global_step

    def _fit(self, X):
        # if self.is_debug: print("Successfully entered _fit")
        self.A_Z = self.localModel.forward(X)
        return self.A_Z

    def _fit_back(self, X, back_grad):
        if self.is_debug: print("backprop succesfull")
        self.localModel.backward(X, back_grad)

    def send_components(self):
        if self.is_debug: print("Step 2: performs forward on the local model")
        return self._fit(self.X)

    def receive_gradients(self, back_grad):
        if self.is_debug: print("Step 11: performs backprop on host")
        self._fit_back(self.X, back_grad)

    def send_predict(self, X):
        if self.is_debug: print("Step 12: Fits prediction")
        return self._fit(X)

class VFLFascilitator(object):

    def __init__(self, part_a_out, part_b_out):
        self.is_debug = True
        self.dense_model_c1 = DenseModel(input_dim=part_b_out, output_dim=1, bias=False)
        self.dense_model_c2 = DenseModel(input_dim=part_a_out, output_dim=1, bias=True)
        # activations of the local model
        self.a_l_c1 = None
        self.a_l_c2 = None
        # activations after the dense model
        self.a_c1 = None
        self.a_c2 = None

    def set_dense_model(self, dense_model,name):
        if name == "Client 1":
            self.dense_model_c1 = dense_model
        elif name == "Client 2":
            self.dense_model_c2 = dense_model

    def receive_activations(self,activations,name):
        if name == "Client 1":
            self.a_l_c1 = activations
        elif name == "Client 2":
            self.a_l_c2 = activations
        return self._fit(name)

    def _fit(self,name):
        if name == "Client 1":
            self.a_c1 = self.dense_model_c1.forward(self.a_l_c1)
            if self.is_debug: print("Step 3a: performs forward on the dense model for Client 1")
            A_U = self.a_c1
        elif name == "Client 2":
            self.a_c2 = self.dense_model_c2.forward(self.a_l_c2)
            if self.is_debug: print("Step 3b: performs forward on the dense model for Client 2")
            A_U = self.a_c2
        return A_U

    def _compute_common_gradient(self):
        if self.is_debug: print("Step 4: adds the activation of both parties")
        U = self.a_c1 + self.a_c2
        return U

    def receive_concatination(self):
        return self._compute_common_gradient()

    def receive_gradients(self, gradients):
        if self.is_debug: print("Step 9: receives gradients")
        self.common_grad = gradients

    def perform_back(self,name):
        back_grad = self._fit_back(name)
        return back_grad

    def _fit_back(self,name):
        if name == "Client 1":
            if self.is_debug: print("Step 10a: performs backprop Client 1")
            back_grad = self.dense_model_c1.backward(self.a_l_c1, self.common_grad)
        elif name == "Client 2":
            if self.is_debug: print("Step 10b: performs backprop Client 2")
            back_grad = self.dense_model_c2.backward(self.a_l_c2, self.common_grad)
        return back_grad