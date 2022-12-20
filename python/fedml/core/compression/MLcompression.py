from __future__ import print_function
import torch
import numpy as np
import time
import math
from scipy import stats

import logging

from . import utils



from fedml.core.compression.constants import (
    NO_COMPRESS, TOPK, EFTOPK, GAUSSIAN, RANDOMK, EFRANDOMK, DGCSAMPLING, QUANTIZE, QSGD)


# SPARSIFICATIONS = ["topk", "eftopk", "gaussian", "randomk", "randomkec", "dgcsampling"]
# QUANTIZATIONS = ["quantize", "qsgd"]
# STATEFUL_COMPRESSORS = ["eftopk", "randomkec", "dgcsampling"]


SPARSIFICATIONS = [TOPK, EFTOPK, GAUSSIAN, RANDOMK, EFRANDOMK, DGCSAMPLING]
QUANTIZATIONS = [QUANTIZE, QSGD]
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
        elif direction == "download":
            sigma_scale = args.down_compression_sigma_scale,
            ratio = args.down_compression_sparse_ratio
            quantize_level = args.down_compression_quantize_level,
            is_biased = args.down_compression_is_biased
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
            s = 2 ** quantize_level - 1
            values = self.get_qsgd(tensor, s, is_biased)
        else:
            values = tensor
        return values

    def decompress_new(self, tensor):
        return tensor

    def update_shapes_dict(self, tensor, name):
        self.shapes[name] = tensor.shape


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


















