import logging
import math
import time
import warnings
from collections import defaultdict, OrderedDict
from copy import deepcopy
from functools import partial
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn.modules.utils import _pair, _quadruple

from .attack_base import BaseAttackMethod
from ..constants import cifar10_mean, cifar10_std

"""
ref: Geiping, Jonas, et al. "Inverting gradients-how easy is it to break privacy in federated learning?." 
Advances in Neural Information Processing Systems 33 (2020): 16937-16947.

attack @ (malicious) server
Steps:
(1) At the very beginning, the malicious server sends initialized parameters to clients and receives the local gradients from one client.
(2) The adversary decomposes a parameter gradient into its norm magnitiude and its direction
(3) The adversary recovery original data of the client via minimizing cosine similarity between gradients of g(x,y) and g(x*, y), 
	which is solved by an Adam solver.
    Specifically, we would like to solve: arg min_x (1 - <g(x,y), g(x*, y)> / ||g(x,y)||* ||g(x*,y)|| + alpha * TV(x)),
    where TV is total variation as a simple image prior to the overall problem. ref: Nonlinear total variation based noise removal algorithms, (https://www.sciencedirect.com/science/article/abs/pii/016727899290242F) 

TODO: add more description abour different settings and variables

"""


class InvertAttack(BaseAttackMethod):
    def __init__(
        self, attack_client_idx=0, trained_model=False, model=None, num_images=1, use_updates=False,
    ):
        defs = ConservativeStrategy()
        loss_fn = Classification()
        self.use_updates = use_updates
        self.img_shape = (3, 32, 32)
        self.model = model
        self.model.eval()
        self.dm = torch.as_tensor(cifar10_mean)[:, None, None]
        self.ds = torch.as_tensor(cifar10_std)[:, None, None]
        self.num_images = num_images  # = batch_size in local training

    def reconstruct_data(self, a_gradient: dict, extra_auxiliary_info: Any = None):
        self.ground_truth = extra_auxiliary_info[0][0]
        self.labels = extra_auxiliary_info[0][1]

        if not self.use_updates:
            rec_machine = GradientReconstructor(
                self.model, (self.dm, self.ds), config=extra_auxiliary_info[1], num_images=self.num_images,
            )
            self.input_gradient = a_gradient
            output, stats = rec_machine.reconstruct(self.input_gradient, self.labels, self.img_shape)
        else:
            rec_machine = FedAvgReconstructor(
                self.model,
                (self.dm, self.ds),
                # self.local_steps,
                # self.local_lr,
                config=extra_auxiliary_info[1],
                use_updates=self.use_updates,
            )
            self.input_parameters = a_gradient
            output, stats = rec_machine.reconstruct(self.input_parameters, self.labels, self.img_shape)

        test_mse = (output.detach() - self.ground_truth).pow(2).mean()
        feat_mse = (self.model(output.detach()) - self.model(self.ground_truth)).pow(2).mean()
        test_psnr = psnr(output, self.ground_truth, factor=1 / self.ds)
        logging.info(
            f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |"
        )


"""
Train Reconstructor for recovering original images from a client
"""

"""Optimization setups."""

from dataclasses import dataclass


@dataclass
# class ConservativeStrategy(Strategy):
class ConservativeStrategy:
    """Default usual parameters, defines a config object."""

    def __init__(self, lr=None, epochs=None, dryrun=False):
        """Initialize training hyperparameters."""
        self.lr = 0.1
        self.epochs = 120
        self.batch_size = 128
        self.optimizer = "SGD"
        self.scheduler = "linear"
        self.warmup = False
        self.weight_decay: float = 5e-4
        self.dropout = 0.0
        self.augmentations = True
        self.dryrun = False
        # super().__init__(lr=None, epochs=None, dryrun=False)


class Loss:
    """Abstract class, containing necessary methods.
    Abstract class to collect information about the 'higher-level' loss function, used to train an energy-based model
    containing the evaluation of the loss function, its gradients w.r.t. to first and second argument and evaluations
    of the actual metric that is targeted.
    """

    def __init__(self):
        """Init."""
        pass

    def __call__(self, reference, argmin):
        """Return l(x, y)."""
        raise NotImplementedError()
        # return value, name, format

    def metric(self, reference, argmin):
        """The actually sought metric."""
        raise NotImplementedError()
        # return value, name, format


class Classification(Loss):
    """A classical NLL loss for classification. Evaluation has the softmax baked in.
    The minimized criterion is cross entropy, the actual metric is total accuracy.
    """

    def __init__(self):
        """Init with torch MSE."""
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=None, size_average=None, ignore_index=-100, reduce=None, reduction="mean",
        )

    def __call__(self, x=None, y=None):
        """Return l(x, y)."""
        name = "CrossEntropy"
        format = "1.5f"
        if x is None:
            return name, format
        else:
            value = self.loss_fn(x, y)
            return value, name, format

    def metric(self, x=None, y=None):
        """The actually sought metric."""
        name = "Accuracy"
        format = "6.2%"
        if x is None:
            return name, format
        else:
            value = (x.data.argmax(dim=1) == y).sum().float() / y.shape[0]
            return value.detach(), name, format


"""Reconstructor setups."""

DEFAULT_CONFIG = dict(
    signed=False,
    boxed=True,
    cost_fn="sim",
    indices="def",
    weights="equal",
    lr=0.1,
    optim="adam",
    restarts=1,
    max_iterations=4800,
    total_variation=1e-1,
    init="randn",
    filter="none",
    lr_decay=True,
    scoring_choice="loss",
)


def _label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None:
            raise ValueError(f"Deprecated key in config dict: {key}!")
    return config


class GradientReconstructor:
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        self.setup = dict(device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        if self.config["scoring_choice"] == "inception":
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
        self.iDLG = True

    def reconstruct(
        self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None,
    ):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()

        stats = defaultdict(list)
        x = self._init_images(img_shape)
        scores = torch.zeros(self.config["restarts"])

        if labels is None:
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                self.reconstruct_label = False
            else:
                # DLG label recovery
                # However this also improves conditioning for some LBFGS cases
                self.reconstruct_label = True

                def loss_fn(pred, labels):
                    labels = torch.nn.functional.softmax(labels, dim=-1)
                    return torch.mean(torch.sum(-labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))

                self.loss_fn = loss_fn
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False

        try:
            for trial in range(self.config["restarts"]):
                x_trial, labels = self._run_trial(x[trial], input_data, labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x[trial] = x_trial
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            print("Trial procedure manually interruped.")
            pass

        # Choose optimal result:
        print("Choosing optimal result ...")
        scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
        optimal_index = torch.argmin(scores)
        print(f"Optimal result score: {scores[optimal_index]:2.4f}")
        stats["opt"] = scores[optimal_index].item()
        x_optimal = x[optimal_index]

        print(f"Total time: {time.time()-start_time}.")
        return x_optimal.detach(), stats

    def _init_images(self, img_shape):
        if self.config["init"] == "randn":
            return torch.randn((self.config["restarts"], self.num_images, *img_shape))
        else:
            raise ValueError()

    def _run_trial(self, x_trial, input_data, labels, dryrun=False):
        x_trial.requires_grad = True
        if self.reconstruct_label:
            output_test = self.model(x_trial)
            labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

            if self.config["optim"] == "adam":
                optimizer = torch.optim.Adam([x_trial, labels], lr=self.config["lr"])
            else:
                raise ValueError()
        else:
            if self.config["optim"] == "adam":
                optimizer = torch.optim.Adam([x_trial], lr=self.config["lr"])
            elif self.config["optim"] == "sgd":  # actually gd
                optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
            elif self.config["optim"] == "LBFGS":
                optimizer = torch.optim.LBFGS([x_trial])
            else:
                raise ValueError()

        max_iterations = self.config["max_iterations"]
        dm, ds = self.mean_std
        if self.config["lr_decay"]:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[max_iterations // 2.667, max_iterations // 1.6, max_iterations // 1.142,],
                gamma=0.1,
            )  # 3/8 5/8 7/8
        try:
            for iteration in range(max_iterations):
                closure = self._gradient_closure(optimizer, x_trial, input_data, labels)
                rec_loss = optimizer.step(closure)
                if self.config["lr_decay"]:
                    scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.config["boxed"]:
                        x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)

                    if (iteration + 1 == max_iterations) or iteration % 500 == 0:
                        print(f"It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.")

                    if (iteration + 1) % 500 == 0:
                        if self.config["filter"] == "none":
                            pass
                        elif self.config["filter"] == "median":
                            x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                        else:
                            raise ValueError()

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass
        return x_trial.detach(), labels

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            rec_loss = reconstruction_costs(
                [gradient],
                input_gradient,
                cost_fn=self.config["cost_fn"],
                indices=self.config["indices"],
                weights=self.config["weights"],
            )

            if self.config["total_variation"] > 0:
                rec_loss += self.config["total_variation"] * total_variation(x_trial)
            rec_loss.backward()
            if self.config["signed"]:
                x_trial.grad.sign_()
            return rec_loss

        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config["scoring_choice"] == "loss":
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            return reconstruction_costs(
                [gradient],
                input_gradient,
                cost_fn=self.config["cost_fn"],
                indices=self.config["indices"],
                weights=self.config["weights"],
            )
        else:
            raise ValueError()


class FedAvgReconstructor(GradientReconstructor):
    """Reconstruct an image from weights after n gradient descent steps."""

    def __init__(
        self,
        model,
        mean_std=(0.0, 1.0),
        local_steps=2,
        local_lr=1e-4,
        config=DEFAULT_CONFIG,
        num_images=1,
        use_updates=True,
        batch_size=0,
    ):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, mean_std, config, num_images)
        self.local_steps = local_steps
        self.local_lr = local_lr
        self.use_updates = use_updates
        self.batch_size = batch_size

    def _gradient_closure(self, optimizer, x_trial, input_parameters, labels):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            parameters = loss_steps(
                self.model,
                x_trial,
                labels,
                loss_fn=self.loss_fn,
                local_steps=self.local_steps,
                lr=self.local_lr,
                use_updates=self.use_updates,
                batch_size=self.batch_size,
            )
            rec_loss = reconstruction_costs(
                [parameters],
                input_parameters,
                cost_fn=self.config["cost_fn"],
                indices=self.config["indices"],
                weights=self.config["weights"],
            )
            if self.config["total_variation"] > 0:
                rec_loss += self.config["total_variation"] * total_variation(x_trial)
            rec_loss.backward()
            if self.config["signed"]:
                x_trial.grad.sign_()
            return rec_loss

        return closure

    def _score_trial(self, x_trial, input_parameters, labels):
        if self.config["scoring_choice"] == "loss":
            self.model.zero_grad()
            parameters = loss_steps(
                self.model,
                x_trial,
                labels,
                loss_fn=self.loss_fn,
                local_steps=self.local_steps,
                lr=self.local_lr,
                use_updates=self.use_updates,
            )
            return reconstruction_costs(
                [parameters],
                input_parameters,
                cost_fn=self.config["cost_fn"],
                indices=self.config["indices"],
                weights=self.config["weights"],
            )


def loss_steps(
    model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4, use_updates=True, batch_size=0,
):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    if use_updates:
        patched_model_origin = deepcopy(patched_model)
    for i in range(local_steps):
        if batch_size == 0:
            outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        else:
            idx = i % (inputs.shape[0] // batch_size)
            outputs = patched_model(inputs[idx * batch_size : (idx + 1) * batch_size], patched_model.parameters,)
            labels_ = labels[idx * batch_size : (idx + 1) * batch_size]
        loss = loss_fn(outputs, labels_).sum()
        grad = torch.autograd.grad(
            loss, patched_model.parameters.values(), retain_graph=True, create_graph=True, only_inputs=True,
        )

        patched_model.parameters = OrderedDict(
            (name, param - lr * grad_part) for ((name, param), grad_part) in zip(patched_model.parameters.items(), grad)
        )

    if use_updates:
        patched_model.parameters = OrderedDict(
            (name, param - param_origin)
            for ((name, param), (name_origin, param_origin)) in zip(
                patched_model.parameters.items(), patched_model_origin.parameters.items(),
            )
        )
    return list(patched_model.parameters.values())


def reconstruction_costs(gradients, input_gradient, cost_fn="l2", indices="def", weights="equal"):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == "def":
        indices = torch.arange(len(input_gradient))
    elif indices == "top10":
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == "linear":
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == "exp":
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        for i in indices:
            if cost_fn == "sim":
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
        if cost_fn == "sim":
            costs = 1 + costs / math.sqrt(pnorm[0]) / math.sqrt(pnorm[1])

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)


"""Utils setups."""


class MetaMonkey(torch.nn.Module):
    """Trace a networks and then replace its module calls with functional calls.
    This allows for backpropagation w.r.t to weights for "normal" PyTorch networks.
    """

    def __init__(self, net):
        """Init with network."""
        super().__init__()
        self.net = net
        self.parameters = OrderedDict(net.named_parameters())

    def forward(self, inputs, parameters=None):
        """Live Patch ... :> ..."""
        # If no parameter dictionary is given, everything is normal
        if parameters is None:
            return self.net(inputs)

        # But if not ...
        param_gen = iter(parameters.values())
        method_pile = []
        counter = 0

        for name, module in self.net.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                ext_weight = next(param_gen)
                if module.bias is not None:
                    ext_bias = next(param_gen)
                else:
                    ext_bias = None

                method_pile.append(module.forward)
                module.forward = partial(
                    F.conv2d,
                    weight=ext_weight,
                    bias=ext_bias,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                )
            elif isinstance(module, torch.nn.BatchNorm2d):
                if module.momentum is None:
                    exponential_average_factor = 0.0
                else:
                    exponential_average_factor = module.momentum

                if module.training and module.track_running_stats:
                    if module.num_batches_tracked is not None:
                        module.num_batches_tracked += 1
                        if module.momentum is None:  # use cumulative moving average
                            exponential_average_factor = 1.0 / float(module.num_batches_tracked)
                        else:  # use exponential moving average
                            exponential_average_factor = module.momentum

                ext_weight = next(param_gen)
                ext_bias = next(param_gen)
                method_pile.append(module.forward)
                module.forward = partial(
                    F.batch_norm,
                    running_mean=module.running_mean,
                    running_var=module.running_var,
                    weight=ext_weight,
                    bias=ext_bias,
                    training=module.training or not module.track_running_stats,
                    momentum=exponential_average_factor,
                    eps=module.eps,
                )

            elif isinstance(module, torch.nn.Linear):
                lin_weights = next(param_gen)
                lin_bias = next(param_gen)
                method_pile.append(module.forward)
                module.forward = partial(F.linear, weight=lin_weights, bias=lin_bias)

            elif next(module.parameters(), None) is None:
                # Pass over modules that do not contain parameters
                pass
            elif isinstance(module, torch.nn.Sequential):
                # Pass containers
                pass
            else:
                # Warn for other containers
                warnings.warn(f"Patching for module {module.__class__} is not implemented.")

        output = self.net(inputs)

        # Undo Patch
        for name, module in self.net.named_modules():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                module.forward = method_pile.pop(0)
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.forward = method_pile.pop(0)
            elif isinstance(module, torch.nn.Linear):
                module.forward = method_pile.pop(0)

        return output


class MedianPool2d(nn.Module):
    """Median pool (usable as median filter when stride=1) module.
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=True):
        """Initialize with kernel_size, stride, padding."""
        super().__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode="reflect")
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


class InceptionScore(torch.nn.Module):
    """Class that manages and returns the inception score of images."""

    def __init__(self, batch_size=32, setup=dict(device=torch.device("cpu"), dtype=torch.float)):
        """Initialize with setup and target inception batch size."""
        super().__init__()
        self.preprocessing = torch.nn.Upsample(size=(299, 299), mode="bilinear", align_corners=False)
        self.model = torchvision.models.inception_v3(pretrained=True).to(**setup)
        self.model.eval()
        self.batch_size = batch_size

    def forward(self, image_batch):
        """Image batch should have dimensions BCHW and should be normalized.
        B should be divisible by self.batch_size.
        """
        B, C, H, W = image_batch.shape
        batches = B // self.batch_size
        scores = []
        for batch in range(batches):
            input = self.preprocessing(image_batch[batch * self.batch_size : (batch + 1) * self.batch_size])
            scores.append(self.model(input))  # pylint: disable=E1102
        prob_yx = torch.nn.functional.softmax(torch.cat(scores, 0), dim=1)
        entropy = torch.where(prob_yx > 0, -prob_yx * prob_yx.log(), torch.zeros_like(prob_yx))
        return entropy.sum()


def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""

    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return 10 * torch.log10(factor ** 2 / mse)
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float("nan"))
        else:
            return img_batch.new_tensor(float("inf"))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy
