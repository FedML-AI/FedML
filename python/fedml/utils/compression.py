from __future__ import print_function
import torch
import numpy as np
import time
import math
from scipy import stats


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



class EFTopKCompressor(TopKCompressor):
    """
    """
    def __init__(self):
        super().__init__()
        self.name = 'eftopk'

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



class QuantizationCompressor(object):
    def __init__(self):
        self.name = 'quant'
        self.residuals = {}
        self.values = {} 
        self.zc = None
        self.current_ratio = 1
        # ===================================
        self.shapes = {}

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




class QSGDCompressor(object):
    def __init__(self):
        self.name = 'qsgd'
        self.residuals = {}
        self.values = {} 
        self.zc = None
        self.current_ratio = 1
        # ===================================
        self.shapes = {}

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
        'no': NoneCompressor,
        None: NoneCompressor,
        'topk': TopKCompressor,
        'eftopk': EFTopKCompressor, #TopK with error-feedback
        'quantize': QuantizationCompressor, # Naive Quantization Compressor
        'qsgd': QSGDCompressor, # QSGD Quantization Compressor
        }


def gen_threshold_from_normal_distribution(p_value, mu, sigma):
    r"""PPF."""
    zvalue = stats.norm.ppf((1-p_value)/2)
    return mu+zvalue*sigma, mu-zvalue*sigma


def test_gaussion_thres():
    set_mean = 0.0; set_std = 0.5
    d = np.random.normal(set_mean, set_std, 10000)
    k2, p = stats.normaltest(d)
    print(p)
    nnz = np.count_nonzero(d)
    mean = np.mean(d)
    std = np.std(d)
    print('size:%d, nnz: %d' % (d.size, nnz))
    print(set_mean, set_std)
    print(mean, std)
    copyd = d.copy()
    thres = 3*std
    d[np.abs(d) < thres] = 0
    pvalue = 1-np.count_nonzero(d)*1.0/d.size
    print('size:%d, p-value: %f' % (d.size, pvalue))
    left_thres, right_thres = gen_threshold_from_normal_distribution(pvalue, mean, std)
    print('real thres:%f, gen thres: %f' % (thres, right_thres))


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

