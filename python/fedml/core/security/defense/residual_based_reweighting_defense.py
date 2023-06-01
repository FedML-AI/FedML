import math
from collections import OrderedDict
import numpy as np
import torch
from .defense_base import BaseDefenseMethod
from typing import Callable, List, Tuple, Dict, Any
from functools import reduce

"""
Attack-Resistant Federated Learning with Residual-based Reweighting
https://arxiv.org/pdf/1912.11464.pdf
todo: Repeated_Median_Shard, irls, 
code refactoring
"""


class ResidualBasedReweightingDefense(BaseDefenseMethod):
    def __init__(self, config):
        if hasattr(config, "lambda_param"):
            self.lambda_param = config.lambda_param
        else:
            self.lambda_param = 2
        if hasattr(config, "thresh"):
            self.thresh = config.thresh
        else:
            self.thresh = 0.1
        self.mode = config.mode

    def defend_before_aggregation(
        self,
        raw_client_grad_list: List[Tuple[float, OrderedDict]],
        extra_auxiliary_info: Any = None,
    ):
        return self.IRLS_other_split_restricted(raw_client_grad_list)

    def IRLS_other_split_restricted(self, raw_client_grad_list):
        reweight_algorithm = median_reweight_algorithm_restricted
        if self.mode == "median":
            reweight_algorithm = median_reweight_algorithm_restricted
        elif self.mode == "theilsen":
            reweight_algorithm = theilsen_reweight_algorithm_restricted
        elif self.mode == "gaussian":
            reweight_algorithm = gaussian_reweight_algorithm_restricted  # in gaussian reweight algorithm, lambda is sigma

        SHARD_SIZE = 2000
        w = [grad for (_, grad) in raw_client_grad_list]
        w_med = w[0]
        reweight_sum = None

        for k in w_med.keys():
            shape = w_med[k].shape
            total_num = reduce(lambda x, y: x * y, shape)
            # y_list = torch.FloatTensor([grad[k] for grad in w])
            y_list = torch.FloatTensor(len(raw_client_grad_list), total_num)
            for i in range(len(w)):
                y_list[i] = torch.reshape(w[i][k], (-1,))
            transposed_y_list = torch.t(y_list)
            y_result = torch.zeros_like(transposed_y_list)

            if total_num < SHARD_SIZE:
                reweight, restricted_y = reweight_algorithm(
                    transposed_y_list, self.lambda_param, self.thresh
                )
                print(reweight.sum(dim=0))
                if reweight_sum is None:
                    reweight_sum = reweight.sum(dim=0)
                else:
                    reweight_sum += reweight.sum(dim=0)
                y_result = restricted_y
            else:
                num_shards = int(math.ceil(total_num / SHARD_SIZE))
                for i in range(num_shards):
                    y = transposed_y_list[i * SHARD_SIZE : (i + 1) * SHARD_SIZE, ...]
                    reweight, restricted_y = reweight_algorithm(
                        y, self.lambda_param, self.thresh
                    )
                    print(reweight.sum(dim=0))
                    reweight_sum += reweight.sum(dim=0)
                    y_result[i * SHARD_SIZE : (i + 1) * SHARD_SIZE, ...] = restricted_y

            # put restricted y back to w
            y_result = torch.t(y_result)
            for i in range(len(w)):
                w[i][k] = y_result[i].reshape(w[i][k].shape)
            # print(reweight_sum)
        reweight_sum = (reweight_sum / reweight_sum.max()) ** 2
        print(f"reweight_sum={reweight_sum}")
        return zip(reweight_sum, w)


def median_reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2.0 / num_models)]).to(y.device)

    y_median = median(y).unsqueeze(1).repeat(1, num_models)
    residual = y - y_median
    M = median(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = (
        reweight * reshaped_std
    )  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh) + y_median * (reweight < thresh)
    return reweight_regulized, restricted_y


def median(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        output = input[..., int((shape[-1] - 1) / 2)]
    else:
        output = (
            input[..., int(shape[-1] / 2 - 1)] + input[..., int(shape[-1] / 2)]
        ) / 2.0
    return output


def theilsen_reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = theilsen(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)

    # calculate H matrix
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)
    X_X = torch.matmul(X.transpose(1, 2), X)
    X_X = torch.matmul(X, torch.inverse(X_X))
    H = torch.matmul(X_X, X.transpose(1, 2))
    diag = torch.eye(num_models).repeat(total_num, 1, 1).to(y.device)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.FloatTensor([LAMBDA * np.sqrt(2.0 / num_models)]).to(y.device)

    beta = torch.cat(
        (
            intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
            slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
        ),
        dim=-1,
    )
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)
    reweight = processed_H / e * torch.max(-K, torch.min(K, e / processed_H))
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = (
        reweight * reshaped_std
    )  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh) + line_y * (reweight < thresh)
    return reweight_regulized, restricted_y


def gaussian_reweight_algorithm_restricted(y, sig, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float)
    X_pure = X_pure.unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1).to(y.device), X_pure), dim=-1)

    beta = torch.cat(
        (
            intercepts.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
            slopes.repeat(num_models, 1).transpose(0, 1).unsqueeze(2),
        ),
        dim=-1,
    )
    line_y = (beta * X).sum(dim=-1)
    residual = y - line_y
    M = median(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0, 1)

    reweight = gaussian_zero_mean(e, sig=sig)
    reweight_std = reweight.std(dim=1)  # its standard deviation
    reshaped_std = torch.t(reweight_std.repeat(num_models, 1))
    reweight_regulized = (
        reweight * reshaped_std
    )  # reweight confidence by its standard deviation

    restricted_y = y * (reweight >= thresh) + line_y * (reweight < thresh)
    return reweight_regulized, restricted_y


def gaussian_zero_mean(x, sig=1):
    return torch.exp(-x * x / (2 * sig * sig))


def repeated_median(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yyj = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models)).to(y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    eps = np.finfo(float).eps
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.Tensor([float("Inf")] * num_models).to(y.device)
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + diag
    slopes, _ = slopes.sort()
    slopes = median(slopes[:, :, :-1])
    slopes = median(slopes)

    # get intercepts (intercept of median)
    yy_median = median(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.Tensor(xx_median).to(y.device)
    intercepts = yy_median - slopes * xx_median

    return slopes, intercepts


def theilsen(y):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y = y.sort()[0]
    yy = y.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyj = yy
    yyi = yyj.transpose(-1, -2)
    xx = torch.FloatTensor(range(num_models))
    xxj = xx.repeat(total_num, num_models, 1)
    eps = np.finfo(float).eps
    xxi = xxj.transpose(-1, -2) + eps

    diag = torch.FloatTensor([float("Inf")] * num_models)
    inf_lower = torch.tril(diag.repeat(num_models, 1), diagonal=0).repeat(
        total_num, 1, 1
    )
    diag = torch.diag(diag).repeat(total_num, 1, 1)

    dividor = xxi - xxj + diag
    slopes = (yyi - yyj) / dividor + inf_lower
    slopes, _ = torch.flatten(slopes, 1, 2).sort()
    raw_slopes = slopes[:, : int(num_models * (num_models - 1) / 2)]
    slopes = median(raw_slopes)

    # get intercepts (intercept of median)
    yy_median = median(y)
    xx_median = [(num_models - 1) / 2.0] * total_num
    xx_median = torch.FloatTensor(xx_median)
    intercepts = yy_median - slopes * xx_median
    return slopes, intercepts
