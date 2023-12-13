import torch
import copy
from collections import OrderedDict
import pickle

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
        # self.fhe_type = None
        self.context_file = None
        self.he_context = None
        self.is_enabled = False

    def is_fhe_enabled(self):
        return self.is_enabled

    def init(self, args):
        if not self.is_enabled:
            return

        import tenseal as fhe_core

        if hasattr(args, "enable_fhe") and args.enable_fhe:
            logging.info(
                ".......init homomorphic encryption......."
            )
            self.total_client_number = int(args.client_num_in_total)
            self.is_enabled = True
            # script_path = os.getcwd()
            # read in he context file
            with open(__file__[:-10] + '/context.pickle', 'rb') as handle:
                self.context_file = pickle.load(handle)

            # load the context into fhe_core
            self.he_context = fhe_core.context_from(self.context_file)

        if hasattr(args, MLEngineBackend.ml_engine_args_flag) and args.ml_engine in [
            MLEngineBackend.ml_engine_backend_tf,
            MLEngineBackend.ml_engine_backend_jax,
            MLEngineBackend.ml_engine_backend_mxnet,
        ]:
            logging.info(
                "FedML-HE is not supported for the machine learning engine: %s. "
                "We will support more engines in the future iteration." % args.ml_engine
            )
            self.is_enabled = False

    def is_fhe_enabled(self):
        return self.is_enabled

    def fhe_enc(self, enc_type, model_params):
        import tenseal as fhe_core

        # transform tensor to encrypted form
        weight_factors = copy.deepcopy(model_params)
        for key in weight_factors.keys():
            weight_factors[key] = torch.flatten(torch.full_like(weight_factors[key], 1 / self.total_client_number))

        if enc_type == 'local':
            np_params = OrderedDict()
            for key in model_params.keys():
                prepared_tensor = (torch.flatten(model_params[key])) * weight_factors[key]
                np_params[key] = fhe_core.plain_tensor(prepared_tensor)

            enc_model_params = OrderedDict()
            for key in np_params.keys():
                enc_model_params[key] = (fhe_core.ckks_vector(self.he_context, np_params[key])).serialize()
            return enc_model_params
        else:
            # not supported in the current version
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
        import tenseal as fhe_core

        # init a template model
        n_clients = len(list_enc_model_parmas)
        temp_sample_number, temp_model_params = list_enc_model_parmas[0]
        enc_global_params = copy.deepcopy(temp_model_params)

        for i in range(n_clients):
            list_enc_model_parmas[i] = list_enc_model_parmas[i][1]
            for key in enc_global_params.keys():
                list_enc_model_parmas[i][key] = fhe_core.ckks_vector_from(self.he_context,
                                                                          list_enc_model_parmas[i][key])

        for key in enc_global_params.keys():
            for i in range(n_clients):
                if i != 0:
                    # temp = list_enc_model_parmas[i][key] * weight_factors[key]
                    temp = list_enc_model_parmas[i][key]
                    list_enc_model_parmas[0][key] = list_enc_model_parmas[0][key] + temp

        for key in enc_global_params.keys():
            list_enc_model_parmas[0][key] = list_enc_model_parmas[0][key].serialize()

        enc_global_params = list_enc_model_parmas[0]
        return enc_global_params

    def fhe_dec(self, template_model_params, enc_model_params):
        import tenseal as fhe_core

        params_shape = OrderedDict()
        for key in template_model_params.keys():
            params_shape[key] = template_model_params[key].size()

        params_tensor = OrderedDict()
        for key in enc_model_params.keys():
            enc_model_params[key] = fhe_core.ckks_vector_from(self.he_context, enc_model_params[key])
            params_tensor[key] = torch.FloatTensor(enc_model_params[key].decrypt())

        for key in params_tensor.keys():
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
