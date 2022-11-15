import SHELFI_FHE as fhe_core
import torch
import copy
from collections import OrderedDict
import numpy as np

import logging
from ..common.ml_engine_backend import MLEngineBackend

class FedMLFHE:
    _fhe_instance = None

    @staticmethod
    def get_instance():
        if FedMLFHE._fhe_instance is None:
            FedMLFHE._fhe_instance = FedMLFHE()
        return FedMLFHE._fhe_instance

    def __init__(self):
        #self.fhe_type = None
        self.fhe_helper = None
        self.is_enabled = False

    def is_fhe_enabled(self):
        return self.is_enabled   

    def init(self, args):
        if hasattr(args, "enable_fhe") and args.enable_fhe:
            logging.info(
                ".......init Fully Homomorphic Encryption......."
            )
            self.is_enabled = True
            # TODO: parse fhe args
            self.fhe_helper = fhe_core.CKKS('ckks', 4096,  52, 'resources/cryptoparams/')
            self.fhe_helper.loadCryptoParams()
            
        if hasattr(args, MLEngineBackend.ml_engine_args_flag) and args.ml_engine in [
            MLEngineBackend.ml_engine_backend_tf,
            MLEngineBackend.ml_engine_backend_jax,
            MLEngineBackend.ml_engine_backend_mxnet,
        ]:
            logging.info(
                "FedMLFHE is not supported for the machine learning engine: %s. "
                "We will support more engines in the future iteration." % args.ml_engine
            )
            self.is_enabled = False

    def is_fhe_enabled(self):
        return self.is_enabled

    def fhe_enc(self, enc_type, model_params):
        # transform tensor to np arrays for encrypted computation
        # TODO: local enc mode
        if enc_type == 'local':
            np_params = OrderedDict()
            for key in model_params.keys():
                np_params[key] = torch.flatten(model_params[key]).numpy()

            enc_model_params = OrderedDict()
            for key in np_params.keys():
                enc_model_params[key] = self.fhe_helper.encrypt(np_params[key])
            return enc_model_params
        else:
            enc_raw_client_model_or_grad_list = []
            for i in range(len(model_params)):
                local_sample_number, local_model_params = model_params[i]
                np_params = OrderedDict()
                for key in local_model_params.keys():
                    np_params[key] = torch.flatten(local_model_params[key]).numpy()

                enc_model_params = OrderedDict()
                for key in np_params.keys():
                    enc_model_params[key] = self.fhe_helper.encrypt(np_params[key])
                enc_raw_client_model_or_grad_list.append((local_sample_number, enc_model_params))
            return enc_raw_client_model_or_grad_list

    def fhe_fedavg(self, list_enc_model_parmas):
        # init an template model
        temp_sample_number, temp_model_params = list_enc_model_parmas[0]
        enc_global_params =  copy.deepcopy(temp_model_params)
        # (current) weighting factors of basic avg
        n_clients = len(list_enc_model_parmas)
        weight_factors = np.full(n_clients, 1/n_clients).tolist()

        for key in enc_global_params.keys():
            temp_client_layer = []
            for i in range(n_clients):
                local_sample_number, local_model_params = list_enc_model_parmas[i]
                temp_client_layer.append(local_model_params[key])
            enc_global_params[key] = self.fhe_helper.computeWeightedAverage(temp_client_layer, weight_factors)
        return enc_global_params

    def fhe_dec(self, template_model_params, enc_model_params): 
        params_shape = OrderedDict()
        tensor_size = OrderedDict()
        for key in template_model_params.keys():
            params_shape[key] = template_model_params[key].size()
            tensor_size[key] = torch.flatten(template_model_params[key]).numpy().size

        dec_np_params = OrderedDict()
        for key in enc_model_params.keys():
            dec_np_params[key] = self.fhe_helper.decrypt(enc_model_params[key], tensor_size[key])
    
 
        params_tensor = OrderedDict()  
        for key in dec_np_params.keys():
            params_tensor[key] = torch.from_numpy(dec_np_params[key])
            # need torch.Size() to tuple
            params_tensor[key] = torch.reshape(params_tensor[key], tuple(list((params_shape[key]))))
        return params_tensor
    # first decrypt, then transform the plaintext (1D numpy array) to tensor

# def fhe_setup(batch_size=4096,  scaling_factor=40, file_loc="resources/cryptoparams/"):
#     fhe_helper = fhe_core.CKKS("ckks", batch_size, scaling_factor, file_loc)
#     fhe_helper.loadCryptoParams()
#     # create fresh: a context and a key pair
#     #fhe_helper.genCryptoContextAndKeyGen()
#     return fhe_helper













