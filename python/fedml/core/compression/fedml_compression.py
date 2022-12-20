import logging
from collections import OrderedDict




from fedml.core.compression.constants import (
    NO_COMPRESS, TOPK, EFTOPK, GAUSSIAN, RANDOMK, EFRANDOMK, DGCSAMPLING, QUANTIZE, QSGD)
from fedml.core.compression.MLcompression import (
    SPARSIFICATIONS, QUANTIZATIONS, STATEFUL_COMPRESSORS, check_args_compress, create_compressor)



class FedMLCompression:
    _upload_compression_instance = None
    _download_compression_instance = None

    @staticmethod
    def get_instance(direction="upload"):
        if direction == "upload":
            if FedMLCompression._upload_compression_instance is None:
                FedMLCompression._upload_compression_instance = FedMLCompression()
            return FedMLCompression._upload_compression_instance
        elif direction == "download":
            if FedMLCompression._download_compression_instance is None:
                FedMLCompression._download_compression_instance = FedMLCompression()
            return FedMLCompression._download_compression_instance
        else:
            raise NotImplementedError

    def __init__(self):
        self.compressor = None

    def add_status(self, args, client_status, direction="upload"):
        if check_args_compress(args, direction) and self.compressor.stateful:
            assert args.local_cache
            self.compressor.add_status(client_status)



    def load_status(self, args, client_status, direction="upload"):
        if check_args_compress(args, direction) and self.compressor.stateful:
            assert args.local_cache
            self.compressor.add_status(client_status)
        return client_status


    def init(self, args, model, direction="upload"):
        self.compressor = create_compressor(args, model.state_dict(), direction)


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
                    self.compressor.compress(
                        self.compressor.flatten(named_parameters[key]), name=key,
                        sigma_scale=sigma_scale, ratio=ratio
                    )
        elif compression_name in QUANTIZATIONS:
            for key in list(named_parameters.keys()):
                logging.debug("named_parameters[key].shape: {}, named_parameters[key].numel(): {}".format(
                    named_parameters[key].shape, named_parameters[key].numel()
                ))
                compressed_named_parameters[key] = \
                    self.compressor.compress(named_parameters[key], name=key,
                        quantize_level=quantize_level, is_biased=is_biased
                    )
        else:
            raise NotImplementedError
        return compressed_named_parameters, params_indexes



    def decompress_named_parameters(self, named_parameters, params_indexes, args):

        compression_name = getattr(args, "compression", "no")
        if compression_name in SPARSIFICATIONS:
            assert params_indexes is not None
            for k in params_indexes.keys():
                # logging.debug("model_params[k]:{}, model_indexes[k]:{}, k:{}".format(
                #     model_params[k], model_indexes[k], k`
                # ))``
                named_parameters[k] = self.compressor.unflatten(
                    self.compressor.decompress_new(named_parameters[k], params_indexes[k], k), k)
        elif compression_name in QUANTIZATIONS:
            for k in named_parameters.keys():
                # logging.debug("model_params[k]:{}, model_indexes[k]:{}, k:{}".format(
                #     model_params[k], model_indexes[k], k
                # ))
                named_parameters[k] = self.compressor.decompress_new(named_parameters[k])
        else:
            raise NotImplementedError
        return named_parameters



























