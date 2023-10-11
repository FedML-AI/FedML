import logging
from collections import OrderedDict
from typing import List, Tuple

from ...core.common.ml_engine_backend import MLEngineBackend


class FedMLAggOperator:
    @staticmethod
    def agg(args, raw_grad_list: List[Tuple[float, OrderedDict]]) -> OrderedDict:
        """
        Aggregate gradients from multiple clients using a federated learning aggregator.

        Args:
            args: A dictionary containing training configuration parameters.
            raw_grad_list (List[Tuple[float, OrderedDict]]): A list of tuples containing
                local sample counts and gradient updates from client models.

        Returns:
            OrderedDict: The aggregated model parameters.
        """
        training_num = 0
        if args.federated_optimizer == "SCAFFOLD":
            for i in range(len(raw_grad_list)):
                local_sample_num, _, _ = raw_grad_list[i]
                training_num += local_sample_num
        elif args.federated_optimizer == "Mime":
            for i in range(len(raw_grad_list)):
                local_sample_num, _, _ = raw_grad_list[i]
                training_num += local_sample_num
        # elif args.federated_optimizer == "FedDyn":
        #     for i in range(len(raw_grad_list)):
        #         local_sample_num, _, _ = raw_grad_list[i]
        #         training_num += local_sample_num
        else:
            for i in range(len(raw_grad_list)):
                local_sample_num, local_model_params = raw_grad_list[i]
                training_num += local_sample_num

        avg_params = model_aggregator(args, raw_grad_list, training_num)
        return avg_params


def torch_aggregator(args, raw_grad_list, training_num):
    """
    Aggregate gradients or parameters from multiple clients using a federated learning aggregator.

    Args:
        args: A dictionary containing training configuration parameters.
        raw_grad_list (List[Union[Tuple[float, OrderedDict], Tuple[float, OrderedDict, OrderedDict]]]):
            A list of tuples containing local sample counts and gradient updates from client models.
            For some optimizers, it also includes an additional tuple element with local gradients.
        training_num (int): The total number of training samples used for aggregation.

    Returns:
        Union[OrderedDict, Tuple[OrderedDict, OrderedDict]]: The aggregated model parameters or a tuple
            containing aggregated model parameters and aggregated local gradients, depending on the optimizer.
    """

    if args.federated_optimizer == "FedAvg":
        (num0, avg_params) = raw_grad_list[0]
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k] * w
    elif args.federated_optimizer == "FedProx":
        (num0, avg_params) = raw_grad_list[0]
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k] * w
    elif args.federated_optimizer == "FedAvg_seq":
        (num0, avg_params) = raw_grad_list[0]
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                if i == 0:
                    avg_params[k] = local_model_params[k]
                else:
                    avg_params[k] += local_model_params[k]
    elif args.federated_optimizer == "FedOpt":
        pass
    elif args.federated_optimizer == "FedNova":
        pass
    elif args.federated_optimizer == "FedDyn":
        (num0, avg_params) = raw_grad_list[0]
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                # w = 1 / args.client_num_per_round
                if i == 0:
                    avg_params[k] = local_model_params[k]
                else:
                    avg_params[k] += local_model_params[k]
        # (num0, avg_params, avg_local_grad) = raw_grad_list[0]
        # assert args.client_num_per_round == len(raw_grad_list)
        # for k in avg_params.keys():
        #     for i in range(0, len(raw_grad_list)):
        #         local_sample_number, local_model_params, local_grad = raw_grad_list[i]
        #         # w = 1 / args.client_num_per_round
        #         w = local_sample_number / training_num
        #         if i == 0:
        #             avg_params[k] = local_model_params[k] * w
        #             avg_local_grad[k] = local_grad[k] * w
        #         else:
        #             avg_params[k] += local_model_params[k] * w
        #             avg_local_grad[k] += local_grad[k] * w

        # for i in range(0, len(raw_grad_list)):
        #     local_sample_number, local_model_params, local_grad = raw_grad_list[i]
        #     w = 1 / args.client_num_per_round
        #     if i == 0:
        #         avg_local_grad = local_grad * w
        #     else:
        #         avg_local_grad += local_grad * w
        # avg_params = (avg_params, avg_local_grad)
    elif args.federated_optimizer == "SCAFFOLD":
        (num0, total_weights_delta, total_c_delta_para) = raw_grad_list[0]
        # avg_params = total_weights_delta
        for k in total_weights_delta.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, weights_delta, c_delta_para = raw_grad_list[i]
                w = local_sample_number / training_num
                # w = local_sample_number / len(raw_grad_list)
                if i == 0:
                    total_weights_delta[k] = weights_delta[k] * w
                    total_c_delta_para[k] = c_delta_para[k]
                else:
                    total_weights_delta[k] += weights_delta[k] * w
                    total_c_delta_para[k] += c_delta_para[k]
            # w_c = 1 / args.client_num_in_total
            w_c = 1 / args.client_num_in_total
            total_weights_delta[k] = weights_delta[k]
            total_c_delta_para[k] = c_delta_para[k] * w_c
        avg_params = (total_weights_delta, total_c_delta_para)
        # logging.info(f"avg_params:{avg_params}. len(avg_params): {len(avg_params)}")
    elif args.federated_optimizer == "Mime":
        (num0, avg_params, avg_local_grad) = raw_grad_list[0]
        assert args.client_num_per_round == len(raw_grad_list)
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params, local_grad = raw_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                    avg_local_grad[k] = local_grad[k] * w
                else:
                    avg_params[k] += local_model_params[k] * w
                    avg_local_grad[k] += local_grad[k] * w
        avg_params = (avg_params, avg_local_grad)
    return avg_params


def tf_aggregator(args, raw_grad_list, training_num):
    """
    Aggregate gradients or parameters from multiple clients using a TensorFlow-based federated learning aggregator.

    Args:
        args: A dictionary containing training configuration parameters.
        raw_grad_list (List[Tuple[float, List[float]]]): A list of tuples containing local sample counts and
            gradient updates from client models.
        training_num (int): The total number of training samples used for aggregation.

    Returns:
        List[float]: The aggregated model parameters.
    """
    (num0, avg_params) = raw_grad_list[0]

    if args.federated_optimizer == "FedAvg":
        for k in range(0, len(avg_params)):
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k] * w
    elif args.federated_optimizer == "FedAvg_seq":
        for k in range(0, len(avg_params)):
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                if i == 0:
                    avg_params[k] = local_model_params[k]
                else:
                    avg_params[k] += local_model_params[k]
    elif args.federated_optimizer == "FedOpt":
        pass

    return avg_params


def jax_aggregator(args, raw_grad_list, training_num):
    """
    Aggregate gradients or parameters from multiple clients using a JAX-based federated learning aggregator.

    Args:
        args: A dictionary containing training configuration parameters.
        raw_grad_list (List[Tuple[float, Dict[str, Dict[str, float]]]]): A list of tuples containing local sample counts
            and gradient updates from client models. Each update is a dictionary containing 'w' and 'b' keys.

    Returns:
        Dict[str, Dict[str, float]]: The aggregated model parameters containing 'w' and 'b' keys.
    """
    (num0, avg_params) = raw_grad_list[0]

    if args.federated_optimizer == "FedAvg":
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    avg_params[k]["w"] = local_model_params[k]["w"] * w
                    avg_params[k]["b"] = local_model_params[k]["b"] * w
                else:
                    avg_params[k]["w"] += local_model_params[k]["w"] * w
                    avg_params[k]["b"] += local_model_params[k]["b"] * w
    elif args.federated_optimizer == "FedAvg_seq":
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                if i == 0:
                    avg_params[k]["b"] = local_model_params[k]["b"]
                    avg_params[k]["w"] = local_model_params[k]["w"]
                else:
                    avg_params[k]["b"] += local_model_params[k]["b"]
                    avg_params[k]["w"] += local_model_params[k]["w"]
    elif args.federated_optimizer == "FedOpt":
        pass

    return avg_params


def mxnet_aggregator(args, raw_grad_list, training_num):
    """
    Aggregate gradients or parameters from multiple clients using a MXNet-based federated learning aggregator.

    Args:
        args: A dictionary containing training configuration parameters.
        raw_grad_list (List[Tuple[float, Dict[str, List[float]]]]): A list of tuples containing local sample counts
            and gradient updates from client models. Each update is a dictionary containing lists of parameters.

    Returns:
        Dict[str, List[float]]: The aggregated model parameters.
    """
    (num0, avg_params) = raw_grad_list[0]

    if args.federated_optimizer == "FedAvg":
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    for j in range(0, len(avg_params[k])):
                        avg_params[k][j] = local_model_params[k][j] * w
                else:
                    for j in range(0, len(avg_params[k])):
                        avg_params[k][j] += local_model_params[k][j] * w
    elif args.federated_optimizer == "FedAvg_seq":
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                if i == 0:
                    for j in range(0, len(avg_params[k])):
                        avg_params[k][j] = local_model_params[k][j]
                else:
                    for j in range(0, len(avg_params[k])):
                        avg_params[k][j] += local_model_params[k][j]
    elif args.federated_optimizer == "FedOpt":
        pass

    return avg_params


def model_aggregator(args, raw_grad_list, training_num):
    """
    Aggregate gradients or parameters from multiple clients using a federated learning aggregator based on the
    specified machine learning engine.

    Args:
        args: A dictionary containing training configuration parameters.
        raw_grad_list (List[Union[Tuple[float, Dict[str, Dict[str, float]]], Tuple[float, Dict[str, List[float]]]]]):
            A list of tuples containing local sample counts and gradient updates from client models. The format of
            updates varies based on the machine learning engine.

    Returns:
        Union[Dict[str, Dict[str, float]], Dict[str, List[float]]]: The aggregated model parameters or gradients
            based on the selected machine learning engine.
    """
    if hasattr(args, MLEngineBackend.ml_engine_args_flag):
        if args.ml_engine == MLEngineBackend.ml_engine_backend_tf:
            return tf_aggregator(args, raw_grad_list, training_num)
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_jax:
            return jax_aggregator(args, raw_grad_list, training_num)
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_mxnet:
            return mxnet_aggregator(args, raw_grad_list, training_num)
        else:
            return torch_aggregator(args, raw_grad_list, training_num)
    else:
        return torch_aggregator(args, raw_grad_list, training_num)
