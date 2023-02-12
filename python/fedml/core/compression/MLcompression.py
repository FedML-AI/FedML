from __future__ import print_function
import torch
import numpy as np
import time
import math
from scipy import stats

import logging

from . import utils


 
from fedml.core.compression.constants import (
   NO_COMPRESS, TOPK, EFTOPK, GAUSSIAN, RANDOMK, EFRANDOMK, DGCSAMPLING, QUANTIZE, QSGD, ATOMO, POWERSGD)


# SPARSIFICATIONS = ["topk", "eftopk", "gaussian", "randomk", "randomkec", "dgcsampling"]
# QUANTIZATIONS = ["quantize", "qsgd"]
# STATEFUL_COMPRESSORS = ["eftopk", "randomkec", "dgcsampling"]


SPARSIFICATIONS = [TOPK, EFTOPK, GAUSSIAN, RANDOMK, EFRANDOMK, DGCSAMPLING]
QUANTIZATIONS = [QUANTIZE, QSGD]
LOWRANK = [ATOMO]
STATEFUL_COMPRESSORS = [EFTOPK, EFRANDOMK, DGCSAMPLING]


def check_args_compress(args, direction="upload"):
    if direction == "upload":
        compress = getattr(args, "compression", None) is not None \
                and getattr(args, "compression", None) != "no"
    elif direction == "download":
        compress = getattr(args, "down_compression", None) is not None \
                and getattr(args, "down_compression", None) != "no"
    else:
        raise NotImplementedError
    return compress



def check_compress_stateful(args, direction="upload"):
    if direction == "upload":
        compress = getattr(args, "compression", None)
    elif direction == "download":
        compress = getattr(args, "down_compression", None)
    else:
        raise NotImplementedError
    return compress in STATEFUL_COMPRESSORS


class NoneCompressor():
    def __init__(self):
        self.name = 'none'

    def compress(self, tensor):
        return tensor, tensor.dtype

    def decompress(self, tensor, ctc):
        z = tensor 
        return z


class TensorBuffer():
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """
    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors = tensors

        self.buffer = torch.cat([t.view(-1) for t in tensors]) # copies
    
    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(*self._tensors[index].shape)

    def __len__(self):
        return len(self._tensors)

    def pack(self, tensors=None):
        # Optional. init already does this.
        if tensors is None:
            tensors = self._tensors
        for tensor, entry in zip(tensors, self):
            entry[:] = tensor

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor[:] = entry

    def nelement(self):
        return self.buffer.nelement()

    def element_size(self):
        return self.buffer.element_size()

    def bits(self):
        return 8 * self.nelement() * self.element_size()


class TopKCompressor():
    """
    Sparse Communication for Distributed Gradient Descent, Alham Fikri Aji et al., 2017
    """
    def __init__(self):
        self.residuals = {}
        self.sparsities = []
        self.zero_conditions = {}
        self.values = {} 
        self.indexes = {} 
        self.c = 0
        self.t = 0.
        self.name = 'topk'
        self.zc = None
        self.current_ratio = 1
        # ===================================
        self.shapes = {}
        self.stateful = False


    def compress_named_parameters(self, named_parameters, args, direction="upload"):
        # compressed_named_parameters, params_indexes = \
        #     compress_named_parameters(named_parameters, self.compressor, self.args.compression,
        #         sigma_scale=self.args.compression_sigma_scale, sparse_ratio=self.args.compression_sparse_ratio,
        #         quantize_level=self.args.compression_quantize_level, is_biased=self.args.compression_is_biased)
        compressed_named_parameters = {}
        params_indexes = {}
        compression_name = getattr(args, "compression", "no")
        if direction == "upload":
            sigma_scale = args.compression_sigma_scale,
            ratio = args.compression_sparse_ratio
            quantize_level = args.compression_quantize_level,
            is_biased = args.compression_is_biased
            factorization_rank = args.factorization_rank
        elif direction == "download":
            sigma_scale = args.down_compression_sigma_scale,
            ratio = args.down_compression_sparse_ratio
            quantize_level = args.down_compression_quantize_level,
            is_biased = args.down_compression_is_biased
            factorization_rank = args.down_factorization_rank
        else:
            raise NotImplementedError

        if compression_name in SPARSIFICATIONS:
            for key in list(named_parameters.keys()):
                logging.debug("named_parameters[key].shape: {}, named_parameters[key].numel(): {}".format(
                    named_parameters[key].shape, named_parameters[key].numel()
                ))
                _, params_indexes[key], compressed_named_parameters[key] = \
                    self.compress(
                        self.flatten(named_parameters[key]), name=key,
                        sigma_scale=sigma_scale, ratio=ratio
                    )
        elif compression_name in QUANTIZATIONS:
            for key in list(named_parameters.keys()):
                logging.debug("named_parameters[key].shape: {}, named_parameters[key].numel(): {}".format(
                    named_parameters[key].shape, named_parameters[key].numel()
                ))
                compressed_named_parameters[key] = \
                    self.compress(named_parameters[key], name=key,
                        quantize_level=quantize_level, is_biased=is_biased
                    )
        elif compression_name in LOWRANK:
            for key in list(named_parameters.keys()):
                logging.debug("named_parameters[key].shape: {}, named_parameters[key].numel(): {}".format(
                    named_parameters[key].shape, named_parameters[key].numel()
                ))
                compressed_named_parameters[key] = \
                    self.compress(named_parameters[key], name=key,
                        factorization_rank=factorization_rank
                    )
        else:
            raise NotImplementedError
        return compressed_named_parameters, params_indexes



    def decompress_named_parameters(self, named_parameters, params_indexes, args):
        # named_parameters = uncompress_named_parameters(named_parameters, params_indexes,
        #                             self.compressor, getattr(self.args, "compression", None))

        compression_name = getattr(args, "compression", "no")
        if compression_name in SPARSIFICATIONS:
            assert params_indexes is not None
            for k in params_indexes.keys():
                # logging.debug("model_params[k]:{}, model_indexes[k]:{}, k:{}".format(
                #     model_params[k], model_indexes[k], k`
                # ))``
                named_parameters[k] = self.unflatten(
                    self.decompress_new(named_parameters[k], params_indexes[k], k), k)
        elif compression_name in QUANTIZATIONS:
            for k in named_parameters.keys():
                # logging.debug("model_params[k]:{}, model_indexes[k]:{}, k:{}".format(
                #     model_params[k], model_indexes[k], k
                # ))
                named_parameters[k] = self.decompress_new(named_parameters[k])
        elif compression_name in LOWRANK:
            for k in named_parameters.keys():
                # logging.debug("model_params[k]:{}, model_indexes[k]:{}, k:{}".format(
                #     model_params[k], model_indexes[k], k
                # ))
                named_parameters[k] = self.decompress(named_parameters[k], k)
        else:
            raise NotImplementedError
        return named_parameters



    def _process_data_before_selecting(self, name, data):
        pass

    def _process_data_after_residual(self, name, data):
        if name not in self.zero_conditions:
            self.zero_conditions[name] = torch.ones(data.numel(), dtype=torch.float32, device=data.device) 
        zero_condition = self.zero_conditions[name]
        zero_condition.fill_(1.0)
        zero_condition[self.indexes[name]] = 0.0
        self.zc = zero_condition

    def clear(self):
        self.residuals = {}
        self.sparsities = []
        self.zero_conditions = {}
        self.values = {} 
        self.indexes = {} 


    def compress(self, tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]

            self.values[name] = values
            self.indexes[name] = indexes

            return tensor, indexes, values

    def decompress(self, tensor, original_tensor_size):
        return tensor


    def decompress_new(self, tensor, indexes, name=None, shape=None):
        '''
            Just decompress, without unflatten.
            Remember to do unflatter after decompress
        '''
        if shape is None:
            decompress_tensor = torch.zeros(
                self.shapes[name], dtype=tensor.dtype, device=tensor.device).view(-1)
            # logging.info(f"decompress_tensor.shape:{decompress_tensor.shape}. \
            #     self.shapes[name]:{self.shapes[name]}, \
            #     indexes.shape: {indexes.shape}, tensor.shape: {tensor.shape}")

            decompress_tensor[indexes] = tensor
            # decompress_tensor = torch.zeros(self.shapes[name]).view(-1)
            # decompress_tensor[indexes] = tensor.type(decompress_tensor.dtype)
            return decompress_tensor
        else:
            decompress_tensor = torch.zeros(
                self.shapes[name], dtype=tensor.dtype, device=tensor.device).view(-1)
            decompress_tensor[indexes] = tensor
            # decompress_tensor = torch.zeros(self.shapes[name]).view(-1)
            # decompress_tensor[indexes] = tensor.type(decompress_tensor.dtype)
            return decompress_tensor

    def flatten(self, tensor, name=None):
        ''' 
            flatten a tensor 
        '''
        self.shapes[name] = tensor.shape
        return tensor.view(-1)

    def unflatten(self, tensor, name=None, shape=None):
        ''' 
            unflatten a tensor 
        '''
        if shape is None:
            return tensor.view(self.shapes[name])
        else:
            return tensor.view(shape)

    def update_shapes_dict(self, tensor, name):
        self.shapes[name] = tensor.shape

    def get_residuals(self, name, like_tensor):
        if name not in self.residuals:
            self.residuals[name] = torch.zeros_like(like_tensor.data)
        return self.residuals[name]

    def add_residuals(self, included_indexes, name):
        with torch.no_grad():
            residuals = self.residuals[name]
            if type(included_indexes) is np.ndarray:
                indexes_t = torch.from_numpy(included_indexes).to(device=residuals.device).long()
            else:
                indexes_t = included_indexes
            values = self.values[name]
            values.data[indexes_t] = 0.0
            residuals.data[self.indexes[name]] += values.data
            #selected_indexes = TopKCompressor.indexes[name][indexes_t]
            #residuals.data[selected_indexes] = 0.0 
            #logger.info('residuals after: %f', torch.norm(TopKCompressor.residuals[name].data))

    def load_status(self, args, client_status):
        if "compression_residuals" not in client_status:
            self.residuals = {}
        else:
            self.residuals = client_status["compression_residuals"]

    def add_status(self, client_status):
        client_status["compression_residuals"] = self.residuals


class EFTopKCompressor(TopKCompressor):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'eftopk'
        self.stateful = True

    def compress(self, tensor, name=None, sigma_scale=2.5, ratio=0.05):
        start = time.time()
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            # top-k solution
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio
            #tensor.data.add_(TopKCompressor.residuals[name].data)
            self._process_data_before_selecting(name, tensor.data)

            values, indexes = torch.topk(torch.abs(tensor.data), k=k)
            values = tensor.data[indexes]

            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0. 
            self.values[name] = values
            self.indexes[name] = indexes

            self._process_data_after_residual(name, tensor.data)

            return tensor, indexes, values

    def _process_data_before_selecting(self, name, data):
        data.add_(self.residuals[name].data)


class GaussianCompressor(TopKCompressor):
    """
    """

    def __init__(self):
        super().__init__()
        self.name = 'gaussian'
        self.iterations = {}
        self.sparsities = []
        self.stateful = True

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            tensor.add_(self.residuals[name].data)

            std = torch.std(tensor)
            mean = torch.mean(tensor)
            left_thres, right_thres = utils.gen_threshold_from_normal_distribution(1-ratio, float(mean), float(std))
            abs_tensor = torch.abs(tensor)
            loops = 0
            while loops < 5:
                one_indexes = abs_tensor > right_thres
                indexes = one_indexes.nonzero().data.squeeze().view(-1)
                if indexes.numel() < 2*k/3:
                    right_thres *= 0.5
                elif indexes.numel() > 4*k/3:
                    right_thres *= 1.5
                else:
                    break
                loops += 1
            indexes = indexes[0:k]
            values = tensor.data[indexes] 
            #print('gaussion vs topk: ', indexes.numel(), k)
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.0

            self.values[name] = values
            self.indexes[name] = indexes
            self._process_data_after_residual(name, tensor)

            return tensor, indexes, values


class RandomKCompressor(TopKCompressor):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'randomk'
        self.counter = 0
        self.stateful = False

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            perm = torch.randperm(numel, device=tensor.device)
            self.counter += 1
            indexes = perm[:k]
            values = tensor.data[indexes] 

            self.values[name] = values
            self.indexes[name] = indexes

            return tensor, indexes, values


class RandomKECCompressor(TopKCompressor):

    def __init__(self):
        super().__init__()
        self.name = 'randomkec'
        self.counter = 0
        self.stateful = True


    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            tensor.add_(self.residuals[name].data)
            perm = torch.randperm(numel, device=tensor.device)
            self.counter += 1
            indexes = perm[:k]
            values = tensor.data[indexes] 
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.0
            return tensor, indexes, values


class DGCSamplingCompressor(TopKCompressor):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'dgcsampling' # Section 5 of the DGC paper, which samples 0.1% to 1% of the gradients to perform topk
        self.stateful = True

    def _process_data_before_selecting(self, name, data):
        super()._process_data_before_selecting(name, data)
        data.add_(self.residuals[name].data)

    def compress(self, tensor, name=None, sigma_scale=3, ratio=0.05):
        with torch.no_grad():
            if name not in self.residuals:
                self.residuals[name] = torch.zeros_like(tensor.data)
            numel = tensor.numel()
            k = max(int(numel * ratio), 1)
            self.current_ratio = ratio

            self._process_data_before_selecting(name, tensor.data)

            abs_values = torch.abs(tensor.data)

            # First step sampling
            perm = torch.randperm(numel, device=tensor.device)
            if ratio >= 0.01:
                fk=k
            else:
                fk = int(numel * 0.01)
            sampled_indexes = perm[0:fk]
            sampled_values = abs_values[sampled_indexes]
            tmpvalues, tmpindexes = torch.topk(sampled_values, k=k)

            thres = tmpvalues[k-1]
            bool_indexes = abs_values > thres
            indexes = bool_indexes.nonzero().data.squeeze().view(-1)
            num_k = len(indexes)
            if num_k > 4*k/3:
                tmpvalues = abs_values[indexes] 
                values, tmpindexes = torch.topk(tmpvalues, k=k)
                indexes = indexes[tmpindexes]

            values = tensor.data[indexes] 
            self.residuals[name].data = tensor.data + 0.0 
            self.residuals[name].data[indexes] = 0.0

            self.values[name] = values
            self.indexes[name] = indexes
            self._process_data_after_residual(name, tensor)

            return tensor, indexes, values



class QuantizationCompressor(TopKCompressor):
    def __init__(self):
        self.name = 'quant'
        self.residuals = {}
        self.values = {} 
        self.zc = None
        self.current_ratio = 1
        # ===================================
        self.shapes = {}
        self.stateful = False

    def get_naive_quantize(self, x, s, is_biased=False):
        norm = x.norm(p=2)
        # calculate the quantization value of tensor `x` at level `log_2 s`.
        level_float = s * x.abs() / norm
        # floor to quantization
        previous_level = torch.floor(level_float)
        return torch.sign(x) * norm * previous_level / s

    def compress(self, tensor, name=None, quantize_level=32, is_biased=True):
        if quantize_level != 32:
            s = 2 ** quantize_level - 1
            values = self.get_naive_quantize(tensor, s, is_biased)
        else:
            values = tensor
        return values

    def decompress_new(self, tensor):
        return tensor

    def update_shapes_dict(self, tensor, name):
        self.shapes[name] = tensor.shape




class QSGDCompressor(TopKCompressor):
    def __init__(self):
        self.name = 'qsgd'
        self.residuals = {}
        self.values = {} 
        self.zc = None
        self.current_ratio = 1
        # ===================================
        self.shapes = {}
        self.stateful = False

    def get_qsgd(self, x, s, is_biased=False):
        norm = x.norm(p=2)
        # calculate the quantization value of tensor `x` at level `log_2 s`.
        level_float = s * x.abs() / norm
        # floor to quantization
        previous_level = torch.floor(level_float)
        # add the stochastic quantization, to preserve the value in expectation.
        is_next_level = (torch.rand_like(x) < (level_float - previous_level)).float()
        new_level = previous_level + is_next_level

        scale = 1
        if is_biased:
            d = x.nelement()
            # Variance bound of QSGD
            scale = 1.0 / (min(d / (s ** 2), math.sqrt(d) / s) + 1.0)
        return scale * torch.sign(x) * norm * new_level / s

    def qsgd_quantize_numpy(self, x, s, is_biased=False):
        """quantize the tensor x in d level on the absolute value coef wise"""
        norm = np.sqrt(np.sum(np.square(x)))
        # calculate the quantization value of tensor `x` at level `log_2 s`.
        level_float = s * np.abs(x) / norm
        # floor to quantization
        previous_level = np.floor(level_float)
        # add the stochastic quantization, to preserve the value in expectation.
        is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
        new_level = previous_level + is_next_level

        scale = 1
        if is_biased:
            d = len(x)
            # Variance bound of QSGD
            scale = 1.0 / (np.minimum(d / s ** 2, np.sqrt(d) / s) + 1.0)
        return scale * np.sign(x) * norm * new_level / s

    def compress(self, tensor, name=None, quantize_level=32, is_biased=True):
        if quantize_level != 32:
            # logging.info(f"quantize_level:{quantize_level}, type(quantize_level):{type(quantize_level)}")
            s = 2 ** quantize_level - 1
            values = self.get_qsgd(tensor, s, is_biased)
        else:
            values = tensor
        return values

    def decompress_new(self, tensor):
        return tensor

    def update_shapes_dict(self, tensor, name):
        self.shapes[name] = tensor.shape



class ATOMOCompressor(TopKCompressor):
    """
    ATOMO: Communication-efficient Learning via Atomic Sparsification NeurIPS 2018.
    """
    def __init__(self):
        self.name = 'atomo'
        self.shapes = {}
        self.stateful = False

    def reshape_to_2d(self, tensor):
        # slightly from the Atomo paper's reshaping strategy
        return tensor.view(self.ori_tensor_size[0], -1)
        # original Atomo paper's samping strategy
        # if tensor.ndimension() == 1:
        #     return tensor.view(tensor.shape[0] // 2, -1)
        # elif all([s == 1 for s in tensor.shape[2:]]):
        #     return tensor.squeeze()
        # else:
        #     return tensor.view((tensor.shape[0] * tensor.shape[1]) // 2, -1)

    def sample_svd(self, s, factorization_rank=0):
        if s[0] < 1e-6:
            return [0], np.array([1.0])
        probs = s / s[0] if factorization_rank == 0 else factorization_rank * s / s.sum()
        for i, p in enumerate(probs):
            if p > 1:
                probs[i]=1
        sampled_idx = []
        sample_probs = []
        for i, p in enumerate(probs):
            #if np.random.rand() < p:
            # random sampling from bernulli distribution
            if np.random.binomial(1, p):
                sampled_idx += [i]
                sample_probs += [p]
        rank_hat = len(sampled_idx)
        if rank_hat == 0:  # or (rank != 0 and np.abs(rank_hat - rank) >= 3):
            return _sample_svd(s, factorization_rank=factorization_rank)
        return np.array(sampled_idx, dtype=int), np.array(sample_probs)

    def svd(self, matrix):
        # return torch.svd(matrix)  # 1-worker batch.reduce time: 0.76103s
        # return self.svd_with_numpy(matrix)  # 1-worker batch.reduce time: > 2s
        return self.svd_on_cpu(matrix)  # 1-worker batch.reduce time: 0.31790s

    def svd_on_cpu(self, matrix):
        u, s, v = torch.svd(matrix.to('cpu'))
        u = u.to(self.device)
        v = v.to(self.device)
        s = s.to(self.device)
        return u, s, v

    def svd_with_numpy(self, matrix):
        u, s, vT = np.linalg.svd(matrix.cpu().numpy())
        u = torch.from_numpy(u).to(self.device)
        s = torch.from_numpy(s).to(self.device)
        v = torch.from_numpy(vT.transpose()).to(self.device)
        return u, s, v

    def compress(self, tensor, name=None, factorization_rank=4):
        """
        Factorize gradients
        :param tensor: torch.Tensor
        :param factorization_rank: int
        """
        self.device = tensor.device
        self.ori_tensor_size = tensor.size()

        if len(self.ori_tensor_size) == 4:
            matrix = self.reshape_to_2d(tensor)
        elif len(self.ori_tensor_size) == 2:
            matrix = tensor.clone()
        else:
            return tensor
            # raise NotImplementedError("Unsupported tensor dim ...")

        # print("** matrix size: {}".format(matrix.size()))
        u, s, v = self.svd(matrix)
        i, probs = self.sample_svd(s, factorization_rank)
        u = u[:, i]
        s = s[i] / probs
        v = v[:, i]
        compressed = {"u":u, "s":s, "v":v}
        return compressed

    def decompress(self, compressed, name=None):
        """
        Factorized gradients
        :param compressed: Dict
        """
        if name is not None:
            if self.shapes[name] is None:
                return compressed
            else:
                u, s, v = compressed["u"], compressed["s"], compressed["v"]
                decompressed = torch.einsum('md, d, nd -> mn', u, s, v)
            return decompressed.view(self.shapes[name])
        else:
            u, s, v = compressed["u"], compressed["s"], compressed["v"]
            decompressed = torch.einsum('md, d, nd -> mn', u, s, v)
            return decompressed.view(self.ori_tensor_size)


    def update_shapes_dict(self, tensor, name):
        if len(tensor.size()) in [4, 2]:
            self.shapes[name] = tensor.size()
        else:
            self.shapes[name] = None


compressors = {
    NO_COMPRESS: NoneCompressor,
    None: NoneCompressor,
    TOPK: TopKCompressor,
    EFTOPK: EFTopKCompressor, #TopK with error-feedback
    GAUSSIAN: GaussianCompressor, #GaussianK with error-feedback
    RANDOMK: RandomKCompressor, #RandomK without error-feedback
    EFRANDOMK: RandomKECCompressor, #RandomK with error-feedback
    DGCSAMPLING: DGCSamplingCompressor, #DGC (doubling sampling) with error-feedback

    QUANTIZE: QuantizationCompressor, # Naive Quantization Compressor
    QSGD: QSGDCompressor, # QSGD Quantization Compressor

    ATOMO: ATOMOCompressor
}

# compressors = {
#     'no': NoneCompressor,
#     None: NoneCompressor,
#     'topk': TopKCompressor,
#     'eftopk': EFTopKCompressor, #TopK with error-feedback
#     'gaussian': GaussianCompressor, #GaussianK with error-feedback
#     'randomk': RandomKCompressor, #RandomK without error-feedback
#     'randomkec': RandomKECCompressor, #RandomK with error-feedback
#     'dgcsampling': DGCSamplingCompressor, #DGC (doubling sampling) with error-feedback

#     'quantize': QuantizationCompressor, # Naive Quantization Compressor
#     'qsgd': QSGDCompressor, # QSGD Quantization Compressor
#     }



def create_compressor(args, model_params, direction="upload"):
    if check_args_compress(args, direction):
        if direction == "upload":
            compressor =  compressors[args.compression]()
            logging.info(f".......init compresser.......{args.compression} - \
                compression_sparse_ratio: {args.compression_sparse_ratio}, \
                compression_quantize_level: {args.compression_quantize_level}")
        elif direction == "download":
            compressor =  compressors[args.down_compression]()
            logging.info(f".......init down_compresser.......{args.down_compression} - \
                down_compression_sparse_ratio: {args.down_compression_sparse_ratio}, \
                down_compression_quantize_level: {args.down_compression_quantize_level}")
        else:
            raise NotImplementedError

        for k in model_params.keys():
            compressor.update_shapes_dict(model_params[k], k)
    else:
        compressor = None
    return compressor




if __name__ == '__main__':
    #test_gaussion_thres()
    compressor_str = 'topk'
    compressor = compressors[compressor_str]()
    z = torch.rand(128, 256)
    compressed_tensor, _, _ = compressor.compress(z)
    print('compressed shape: ', compressed_tensor.shape)
    decompressed_tensor = compressor.decompress(compressed_tensor, z.size())
    print('decompressed shape: ', decompressed_tensor.shape)
    diff = (decompressed_tensor - z).norm()
    print('difff norm: ', diff)

    compressor_str = 'atomo'
    compressor = compressors[compressor_str]()
    z = torch.rand(128, 256, 3, 3)
    for fr in (4, 8, 16, 32, 64, 128):
        compressed = compressor.compress(z, factorization_rank=fr)
        print("SVD Rank: {}, u shape: {}, s shape: {}, v shape: {}".format(
                                                            fr,
                                                            compressed["u"].size(),
                                                            compressed["s"].size(),
                                                            compressed["v"].size()))
        decompressed_tensor = compressor.decompress(compressed)
        print("decompressed tensor size: {}".format(decompressed_tensor.size()))
        diff = (decompressed_tensor - z).norm()
        print('SVD Rank: {}, difff norm: {}\n'.format(fr, diff))


















