import logging
from collections import OrderedDict




from fedml.core.compression.constants import (
    NO_COMPRESS, TOPK, EFTOPK, GAUSSIAN, RANDOMK, EFRANDOMK, DGCSAMPLING, QUANTIZE, QSGD)
from fedml.core.compression.MLcompression import check_args_compress, create_compressor




class FedMLCompression:
    _compression_instance = None

    @staticmethod
    def get_instance():
        if FedMLCompression._compression_instance is None:
            FedMLCompression._compression_instance = FedMLCompression()
        return FedMLCompression._compression_instance

    def __init__(self):
        self.compressor = None


    def add_status(self, args, client_status):
        if check_args_compress(args) and self.compressor.stateful:
            assert args.local_cache
            self.compressor.add_status(client_status)



    def load_status(self, args, client_status):
        if check_args_compress(args) and self.compressor.stateful:
            assert args.local_cache
            self.compressor.add_status(client_status)
        return client_status


    def init(self, args, model):
        if check_args_compress(args):
            logging.info(f".......init compresser.......{args.compression} - \
                compression_sparse_ratio: {args.compression_sparse_ratio}, \
                compression_quantize_level: {args.compression_quantize_level}")
            self.compressor = create_compressor(args, model.state_dict())
        else:
            self.compressor = None


    def compress(self):
        pass
        # self.compressor



    def decompress(self):
        pass
        # self.compressor





























