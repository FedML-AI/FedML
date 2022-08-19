import logging

from .utils import transform_tensor_to_list

from ....utils.compression import compressors
from ....utils.model_utils import average_named_params, get_average_weight


class FedLocalSGDTrainer(object):
    def __init__(
        self,
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        train_data_num,
        device,
        args,
        model_trainer,
    ):
        self.trainer = model_trainer

        self.client_index = client_index
        self.train_data_local_dict = train_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict
        self.all_train_data_num = train_data_num
        # self.train_local = self.train_data_local_dict[client_index]
        # self.local_sample_number = self.train_data_local_num_dict[client_index]

        self.device = device
        self.args = args
        # ==================================================
        self.compressor = compressors[args.compression]()


    def get_model_params(self):
        weights = self.trainer.get_model_params()
        if self.args.compression is None or self.args.compression == 'no':
            compressed_weights = weights
            model_indexes = None

        elif self.args.compression in ['topk', 'randomk', 'gtopk', 'randomkec', 'eftopk', 'gtopkef']:
            compressed_weights = {}
            model_indexes = {}
            for key in list(weights.keys()):
                logging.debug("weights[key].shape: {}, weights[key].numel(): {}".format(
                    weights[key].shape, weights[key].numel()
                ))
                _, model_indexes[key], compressed_weights[key] = \
                    self.compressor.compress(
                        self.compressor.flatten(weights[key]), name=key,
                        sigma_scale=3, ratio=self.args.compress_ratio
                    )
        elif self.args.compression in ['quantize', 'qsgd', 'sign']:
            compressed_weights = {}
            model_indexes = None
            for key in list(weights.keys()):
                logging.debug("weights[key].shape: {}, weights[key].numel(): {}".format(
                    weights[key].shape, weights[key].numel()
                ))
                compressed_weights[key] = self.compressor.compress(
                    weights[key], name=key,
                    quantize_level=self.args.quantize_level, is_biased=self.args.is_biased
                )
        else:
            raise NotImplementedError

        return compressed_weights, model_indexes


    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def update_dataset(self, client_index):
        self.client_index = client_index
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]

    def train(self, round_idx=None):
        self.args.round_idx = round_idx
        self.trainer.train_iterations(self.train_local, self.device, self.args)

        compressed_weights, model_indexes = self.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_tensor_to_list(compressed_weights)
        return compressed_weights, model_indexes, self.local_sample_number














