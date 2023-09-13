import logging
from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):
    """
    The Architect class is responsible for architecture optimization in neural architecture search (NAS).
    It adapts the architecture of a neural network to improve its performance on a specific task using gradient-based methods.

    Attributes:
        network_momentum (float): The momentum term for the network weights.
        network_weight_decay (float): The weight decay term for the network weights.
        model (nn.Module): The neural network model for which the architecture is optimized.
        criterion (nn.Module): The loss criterion used for training.
        optimizer (torch.optim.Optimizer): The optimizer for architecture parameters.
        device (torch.device): The device on which the operations are performed.
        is_multi_gpu (bool): Flag indicating if the model is trained on multiple GPUs.

    Args:
        model (nn.Module): The neural network model being optimized.
        criterion (nn.Module): The loss criterion for training.
        args (object): A configuration object containing hyperparameters.
        device (torch.device): The device (e.g., 'cuda' or 'cpu') on which to perform computations.

    Methods:
        step(input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
            Perform a single step of architecture optimization.

        step_v2(input_train, target_train, input_valid, target_valid, lambda_train_regularizer, lambda_valid_regularizer):
            Perform a single step of architecture optimization with custom regularization terms.

        step_single_level(input_train, target_train):
            Perform a single step of architecture optimization for a single level.

        step_wa(input_train, target_train, input_valid, target_valid, lambda_regularizer):
            Perform a single step of architecture optimization with weight adaptation.

        step_AOS(input_train, target_train, input_valid, target_valid):
            Perform a single step of architecture optimization using the AOS method.

        _backward_step(input_valid, target_valid):
            Perform the backward step during optimization.

        _backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer):
            Perform the unrolled backward step during optimization.

        _construct_model_from_theta(theta):
            Construct a new model using architecture parameters.

        _hessian_vector_product(vector, input, target, r=1e-2):
            Compute the product of the Hessian matrix and a vector.

        _compute_unrolled_model(input, target, eta, network_optimizer):
            Compute the unrolled model with updated weights.

    Example:
        # Create an Architect instance
        architect = Architect(model, criterion, args, device)

        # Perform architecture optimization
        architect.step(input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled=True)
    """
    
    def __init__(self, model, criterion, args, device):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.criterion = criterion

        arch_parameters = self.model.arch_parameters()
        self.optimizer = torch.optim.Adam(
            arch_parameters,
            lr=args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay,
        )

        self.device = device

        self.is_multi_gpu = False

    # Momentum: https://blog.paperspace.com/intro-to-optimization-momentum-rmsprop-adam/
    # V_j = coefficient_momentum * V_j - learning_rate * gradient
    # W_j = V_j + W_jx  x
    # https://www.youtube.com/watch?v=k8fTYJPd3_I
    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        """
        Compute the unrolled model with respect to the architecture parameters.

        Args:
            input: Input data.
            target: Target data.
            eta (float): Learning rate.
            network_optimizer: The network optimizer.

        Returns:
            unrolled_model: The unrolled model.
        """
        logits = self.model(input)
        loss = self.criterion(logits, target)   # pylint: disable=E1102

        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(
                network_optimizer.state[v]["momentum_buffer"]
                for v in self.model.parameters()
            ).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = (
            _concat(torch.autograd.grad(loss, self.model.parameters())).data
            + self.network_weight_decay * theta
        )
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment + dtheta)
        )
        return unrolled_model

    # DARTS
    def step(
        self,
        input_train,
        target_train,
        input_valid,
        target_valid,
        eta,
        network_optimizer,
        unrolled,
    ):
        """
        Perform one optimization step for architecture search.

        Args:
            input_train: Training input data.
            target_train: Training target data.
            input_valid: Validation input data.
            target_valid: Validation target data.
            eta (float): Learning rate.
            network_optimizer: The network optimizer.
            unrolled (bool): Whether to compute an unrolled model.
        """
        self.optimizer.zero_grad()
        if unrolled:
            # logging.info("first order")
            self._backward_step_unrolled(
                input_train,
                target_train,
                input_valid,
                target_valid,
                eta,
                network_optimizer,
            )
        else:
            # logging.info("second order")
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    # ours
    def step_v2(
        self,
        input_train,
        target_train,
        input_valid,
        target_valid,
        lambda_train_regularizer,
        lambda_valid_regularizer,
    ):
        """
        Perform one optimization step for architecture search (variant 2).

        Args:
            input_train: Training input data.
            target_train: Training target data.
            input_valid: Validation input data.
            target_valid: Validation target data.
            lambda_train_regularizer (float): Regularization weight for training.
            lambda_valid_regularizer (float): Regularization weight for validation.
        """
        self.optimizer.zero_grad()

        # grads_alpha_with_train_dataset
        logits = self.model(input_train)
        loss_train = self.criterion(logits, target_train)  # pylint: disable=E1102

        arch_parameters = self.model.arch_parameters()
        grads_alpha_with_train_dataset = torch.autograd.grad(
            loss_train, arch_parameters
        )

        self.optimizer.zero_grad()

        # grads_alpha_with_val_dataset
        logits = self.model(input_valid)
        loss_val = self.criterion(logits, target_valid)  # pylint: disable=E1102

        arch_parameters = self.model.arch_parameters()
        grads_alpha_with_val_dataset = torch.autograd.grad(loss_val, arch_parameters)

        # for g_train, g_val in zip(grads_alpha_with_train_dataset, grads_alpha_with_val_dataset):
        #     g_val.data.copy_(lambda_valid_regularizer * g_val.data)
        #     g_val.data.add_(g_train.data.mul(lambda_train_regularizer))

        for g_train, g_val in zip(
            grads_alpha_with_train_dataset, grads_alpha_with_val_dataset
        ):
            temp = g_train.data.mul(lambda_train_regularizer)
            g_val.data.add_(temp)

        arch_parameters = self.model.arch_parameters()
        for v, g in zip(arch_parameters, grads_alpha_with_val_dataset):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        # arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()

        # p_sum = np.sum(
        #     [torch.sum(torch.abs(p)).cpu().detach().numpy() for p in arch_parameters if p.requires_grad])
        # # logging.info("BEFORE step params = %s" % str(p_sum))

        self.optimizer.step()

        # arch_parameters = self.model.module.arch_parameters() if self.is_multi_gpu else self.model.arch_parameters()
        # p_sum = np.sum(
        #     [torch.sum(torch.abs(p)).cpu().detach().numpy() for p in arch_parameters if p.requires_grad])
        # logging.info("AFTER step params = %s" % str(p_sum))

    # ours
    def step_single_level(self, input_train, target_train):
        """
        Perform one optimization step for architecture search (single level).

        Args:
            input_train: Training input data.
            target_train: Training target data.
        """
        self.optimizer.zero_grad()

        # grads_alpha_with_train_dataset
        logits = self.model(input_train)
        loss_train = self.criterion(logits, target_train)  # pylint: disable=E1102

        arch_parameters = (
            self.model.module.arch_parameters()
            if self.is_multi_gpu
            else self.model.arch_parameters()
        )
        grads_alpha_with_train_dataset = torch.autograd.grad(
            loss_train, arch_parameters
        )

        arch_parameters = (
            self.model.module.arch_parameters()
            if self.is_multi_gpu
            else self.model.arch_parameters()
        )
        for v, g in zip(arch_parameters, grads_alpha_with_train_dataset):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

    def step_wa(
        self, input_train, target_train, input_valid, target_valid, lambda_regularizer
    ):
        """
        Perform one optimization step for architecture search (weighted average).

        Args:
            input_train: Training input data.
            target_train: Training target data.
            input_valid: Validation input data.
            target_valid: Validation target data.
            lambda_regularizer (float): Regularization weight.
        """
        self.optimizer.zero_grad()

        # grads_alpha_with_train_dataset
        logits = self.model(input_train)
        loss_train = self.criterion(logits, target_train)  # pylint: disable=E1102

        arch_parameters = (
            self.model.module.arch_parameters()
            if self.is_multi_gpu
            else self.model.arch_parameters()
        )
        grads_alpha_with_train_dataset = torch.autograd.grad(
            loss_train, arch_parameters
        )

        # grads_alpha_with_val_dataset
        logits = self.model(input_valid)
        loss_val = self.criterion(logits, target_valid)  # pylint: disable=E1102

        arch_parameters = (
            self.model.module.arch_parameters()
            if self.is_multi_gpu
            else self.model.arch_parameters()
        )
        grads_alpha_with_val_dataset = torch.autograd.grad(loss_val, arch_parameters)

        for g_train, g_val in zip(
            grads_alpha_with_train_dataset, grads_alpha_with_val_dataset
        ):
            temp = g_train.data.mul(lambda_regularizer)
            g_val.data.add_(temp)

        arch_parameters = (
            self.model.module.arch_parameters()
            if self.is_multi_gpu
            else self.model.arch_parameters()
        )
        for v, g in zip(arch_parameters, grads_alpha_with_val_dataset):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

    def step_AOS(self, input_train, target_train, input_valid, target_valid):
        """
        Perform one optimization step for architecture search (AOS).

        Args:
            input_train: Training input data.
            target_train: Training target data.
            input_valid: Validation input data.
            target_valid: Validation target data.
        """
        self.optimizer.zero_grad()
        output_search = self.model(input_valid)
        arch_loss = self.criterion(output_search, target_valid)  # pylint: disable=E1102
        arch_loss.backward()
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        """
        Perform a backward step for the architecture optimization.

        Args:
            input_valid: Validation input data.
            target_valid: Validation target data.
        """
        logits = self.model(input_valid)
        loss = self.criterion(logits, target_valid)  # pylint: disable=E1102

        loss.backward()

    def _backward_step_unrolled(
        self,
        input_train,
        target_train,
        input_valid,
        target_valid,
        eta,
        network_optimizer,
    ):
        """
        Perform a backward step for the architecture optimization with unrolled training.

        Args:
            input_train: Training input data.
            target_train: Training target data.
            input_valid: Validation input data.
            target_valid: Validation target data.
            eta: Learning rate for unrolled training.
            network_optimizer: The optimizer for the network weights.
        """
        # calculate w' in equation (7):
        # approximate w(*) by adapting w using only a single training step and enable momentum.
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer
        )

        logits = unrolled_model(input_valid)
        unrolled_loss = self.criterion(logits, target_valid)  # pylint: disable=E1102
        unrolled_loss.backward()  # w, alpha

        # the first term of equation (7)
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]

        # vector is (gradient of w' on validation dataset)
        vector = [v.grad.data for v in unrolled_model.parameters()]

        # equation (8) = 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        # equation (7)
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        arch_parameters = (
            self.model.module.arch_parameters()
            if self.is_multi_gpu
            else self.model.arch_parameters()
        )

        for v, g in zip(arch_parameters, dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        """
        Construct a new model from the given theta.

        Args:
            theta: A flattened parameter tensor.

        Returns:
            model_new: A new model constructed using the provided theta.
        """
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        named_parameters = (
            self.model.module.named_parameters()
            if self.is_multi_gpu
            else self.model.named_parameters()
        )
        for k, v in named_parameters:
            v_length = np.prod(v.size())
            params[k] = theta[offset : offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)

        if self.is_multi_gpu:
            new_state_dict = OrderedDict()
            for k, v in model_dict.items():
                logging.info("multi-gpu")
                logging.info("k = %s, v = %s" % (k, v))
                if "module" not in k:
                    k = "module." + k
                else:
                    k = k.replace("features.module.", "module.features.")
                new_state_dict[k] = v
        else:
            new_state_dict = model_dict

        model_new.load_state_dict(new_state_dict)
        return model_new.to(self.device)

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        """
        Calculate the Hessian-vector product.

        Args:
            vector: A list of gradient vectors.
            input: Input data.
            target: Target data.
            r: Regularization term.

        Returns:
            List of Hessian-vector products.
        """
        # vector is (gradient of w' on validation dataset)
        R = r / _concat(vector).norm()
        parameters = (
            self.model.module.parameters()
            if self.is_multi_gpu
            else self.model.parameters()
        )
        for p, v in zip(parameters, vector):
            p.data.add_(R, v)  # w+ in equation (8) # inplace operation

        # get alpha gradient based on w+ in training dataset
        logits = self.model(input)
        loss = self.criterion(logits, target)  # pylint: disable=E1102

        arch_parameters = (
            self.model.module.arch_parameters()
            if self.is_multi_gpu
            else self.model.arch_parameters()
        )
        grads_p = torch.autograd.grad(loss, arch_parameters)

        parameters = (
            self.model.module.parameters()
            if self.is_multi_gpu
            else self.model.parameters()
        )
        for p, v in zip(parameters, vector):
            p.data.sub_(2 * R, v)  # w- in equation (8)

        # get alpha gradient based on w- in training dataset
        logits = self.model(input)
        loss = self.criterion(logits, target)  # pylint: disable=E1102

        arch_parameters = (
            self.model.module.arch_parameters()
            if self.is_multi_gpu
            else self.model.arch_parameters()
        )
        grads_n = torch.autograd.grad(loss, arch_parameters)

        # restore w- to w
        parameters = (
            self.model.module.parameters()
            if self.is_multi_gpu
            else self.model.parameters()
        )
        for p, v in zip(parameters, vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]

    # DARTS
    def step_v2_2ndorder(
        self,
        input_train,
        target_train,
        input_valid,
        target_valid,
        eta,
        network_optimizer,
        lambda_train_regularizer,
        lambda_valid_regularizer,
    ):
        """
        Perform a step for architecture optimization using the second-order method.

        Args:
            input_train: Training input data.
            target_train: Training target data.
            input_valid: Validation input data.
            target_valid: Validation target data.
            eta: Learning rate for unrolled training.
            network_optimizer: The optimizer for the network weights.
            lambda_train_regularizer: Regularization term for training dataset.
            lambda_valid_regularizer: Regularization term for validation dataset.
        """
        self.optimizer.zero_grad()

        # approximate w(*) by adapting w using only a single training step and enable momentum.
        # w has been updated to w'
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer
        )
        # print("BEFORE:" + str(unrolled_model.parameters()))

        """(7)"""
        logits_val = unrolled_model(input_valid)
        valid_loss = self.criterion(logits_val, target_valid)  # pylint: disable=E1102
        valid_loss.backward()  # w, alpha

        # the 1st term of equation (7)
        grad_alpha_wrt_val_on_w_prime = [
            v.grad for v in unrolled_model.arch_parameters()
        ]

        # vector is (gradient of w' on validation dataset)
        grad_w_wrt_val_on_w_prime = [v.grad.data for v in unrolled_model.parameters()]

        # the 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(
            grad_w_wrt_val_on_w_prime, input_train, target_train
        )

        # equation (7)
        for g, ig in zip(grad_alpha_wrt_val_on_w_prime, implicit_grads):
            g.data.sub_(eta, ig.data)

        grad_alpha_term = unrolled_model.new_arch_parameters()
        for g_new, g in zip(grad_alpha_term, grad_alpha_wrt_val_on_w_prime):
            g_new.data.copy_(g.data)

        """(8)"""
        # unrolled_model_train = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_model.zero_grad()

        logits_train = unrolled_model(input_train)
        train_loss = self.criterion(logits_train, target_train)  # pylint: disable=E1102
        train_loss.backward()  # w, alpha

        # the 1st term of equation (8)
        grad_alpha_wrt_train_on_w_prime = [
            v.grad for v in unrolled_model.arch_parameters()
        ]

        # vector is (gradient of w' on validation dataset)
        grad_w_wrt_train_on_w_prime = [v.grad.data for v in unrolled_model.parameters()]

        # the 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(
            grad_w_wrt_train_on_w_prime, input_train, target_train
        )

        # equation (7)
        for g, ig in zip(grad_alpha_wrt_train_on_w_prime, implicit_grads):
            g.data.sub_(eta, ig.data)

        for g_train, g_val in zip(grad_alpha_wrt_train_on_w_prime, grad_alpha_term):
            # g_val.data.copy_(lambda_valid_regularizer * g_val.data)
            # g_val.data.add_(g_train.data.mul(lambda_train_regularizer))
            temp = g_train.data.mul(lambda_train_regularizer)
            g_val.data.add_(temp)

        arch_parameters = (
            self.model.module.arch_parameters()
            if self.is_multi_gpu
            else self.model.arch_parameters()
        )
        for v, g in zip(arch_parameters, grad_alpha_term):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

    # DARTS
    def step_v2_2ndorder2(
        self,
        input_train,
        target_train,
        input_valid,
        target_valid,
        eta,
        network_optimizer,
        lambda_train_regularizer,
        lambda_valid_regularizer,
    ):
        """
        Perform a step for architecture optimization using the second-order method with modifications.

        Args:
            input_train: Training input data.
            target_train: Training target data.
            input_valid: Validation input data.
            target_valid: Validation target data.
            eta: Learning rate for unrolled training.
            network_optimizer: The optimizer for the network weights.
            lambda_train_regularizer: Regularization term for training dataset.
            lambda_valid_regularizer: Regularization term for validation dataset.
        """
         
        self.optimizer.zero_grad()

        # approximate w(*) by adapting w using only a single training step and enable momentum.
        # w has been updated to w'
        unrolled_model = self._compute_unrolled_model(
            input_train, target_train, eta, network_optimizer
        )
        # print("BEFORE:" + str(unrolled_model.parameters()))

        """(7)"""
        logits_val = unrolled_model(input_valid)
        valid_loss = self.criterion(logits_val, target_valid)  # pylint: disable=E1102
        valid_loss.backward()  # w, alpha

        # the 1st term of equation (7)
        grad_alpha_wrt_val_on_w_prime = [
            v.grad for v in unrolled_model.arch_parameters()
        ]

        # vector is (gradient of w' on validation dataset)
        grad_w_wrt_val_on_w_prime = [v.grad.data for v in unrolled_model.parameters()]

        # the 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(
            grad_w_wrt_val_on_w_prime, input_valid, target_valid
        )

        # equation (7)
        for g, ig in zip(grad_alpha_wrt_val_on_w_prime, implicit_grads):
            g.data.sub_(eta, ig.data)

        grad_alpha_term = unrolled_model.new_arch_parameters()
        for g_new, g in zip(grad_alpha_term, grad_alpha_wrt_val_on_w_prime):
            g_new.data.copy_(g.data)

        """(8)"""
        # unrolled_model_train = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_model.zero_grad()

        logits_train = unrolled_model(input_train)
        train_loss = self.criterion(logits_train, target_train)  # pylint: disable=E1102
        train_loss.backward()  # w, alpha

        # the 1st term of equation (8)
        grad_alpha_wrt_train_on_w_prime = [
            v.grad for v in unrolled_model.arch_parameters()
        ]

        # vector is (gradient of w' on validation dataset)
        grad_w_wrt_train_on_w_prime = [v.grad.data for v in unrolled_model.parameters()]

        # the 2nd term of equation (7)
        implicit_grads = self._hessian_vector_product(
            grad_w_wrt_train_on_w_prime, input_train, target_train
        )

        # equation (7)
        for g, ig in zip(grad_alpha_wrt_train_on_w_prime, implicit_grads):
            g.data.sub_(eta, ig.data)

        for g_train, g_val in zip(grad_alpha_wrt_train_on_w_prime, grad_alpha_term):
            g_val.data.copy_(lambda_valid_regularizer * g_val.data)
            g_val.data.add_(g_train.data.mul(lambda_train_regularizer))

        arch_parameters = (
            self.model.module.arch_parameters()
            if self.is_multi_gpu
            else self.model.arch_parameters()
        )
        for v, g in zip(arch_parameters, grad_alpha_term):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()
