import logging
from typing import List, Tuple, Dict

from ...core.common.ml_engine_backend import MLEngineBackend


class FedMLAggOperator:

    @staticmethod
    def agg(args, raw_grad_list: List[Tuple[float, Dict]], op=None) -> Dict:
        training_num = 0
        for i in range(len(raw_grad_list)):
            local_sample_num, local_model_params = raw_grad_list[i]
            # logging.info(f"local_sample_num:{local_sample_num} type(local_sample_num): {type(local_sample_num):}")
            training_num += local_sample_num

        avg_params = model_aggregator(args, raw_grad_list, training_num, op)
        return avg_params

    @staticmethod
    def agg_with_weight(args, params_list: List[Tuple[float, Dict]], training_num, op=None) -> Dict:
        avg_params = model_aggregator(args, params_list, training_num, op)
        return avg_params

    @staticmethod
    def agg_seq(args, agg_params: Dict, new_params: Dict, agg_weight, avg_weight, op=None) -> Dict:
        if op is None:
            if hasattr(args, "agg_operator"):
                op = args.agg_operator
            else:
                op = "weighted_avg"
        else:
            op = op

        if op == "weighted_avg":
            agg_weight += avg_weight
            w = avg_weight
        elif op == "avg":
            agg_weight += 1.0
            w = 1.0
        elif op == "sum":
            w = 1.0

        if agg_params is None or len(agg_params) == 0:
            agg_params = {}
            for k in new_params.keys():
                agg_params[k] = new_params[k] * w
        else:
            for k in agg_params.keys():
                agg_params[k] += new_params[k] * w
        return agg_weight, agg_params


    @staticmethod
    def end_agg_seq(args, agg_params: Dict, agg_weight, op=None):
        if op == "weighted_avg":
            # sum_weight = agg_weight
            for k in agg_params.keys():
                agg_params[k] = agg_params[k] / agg_weight
        elif op == "avg":
            for k in agg_params.keys():
                agg_params[k] = agg_params[k] / agg_weight
        elif op == "sum":
            pass
        return agg_params

    # @staticmethod
    # def agg_seq(args, agg_params: Dict, new_params: Dict, agg_weight, avg_weight, scale_keep_precision, op=None) -> Dict:
    #     if op is None:
    #         if hasattr(args, "agg_operator"):
    #             op = args.agg_operator
    #         else:
    #             op = "weighted_avg"
    #     else:
    #         op = op

    #     if op == "weighted_avg":
    #         agg_weight += avg_weight
    #         w = avg_weight * scale_keep_precision
    #     elif op == "avg":
    #         agg_weight += 1.0
    #         w = 1.0
    #     elif op == "sum":
    #         w = 1.0

    #     if agg_params is None or len(agg_params) == 0:
    #         agg_params = {}
    #         for k in new_params.keys():
    #             agg_params[k] = new_params[k] * w
    #     else:
    #         for k in agg_params.keys():
    #             agg_params[k] += new_params[k] * w
    #     return agg_weight, agg_params


    # @staticmethod
    # def end_agg_seq(args, agg_params: Dict, agg_weight, scale_keep_precision, op=None)
    #     if agg_end:
    #         if op == "weighted_avg":
    #             sum_weight = agg_weight * scale_keep_precision
    #             for k in agg_params.keys():
    #                 agg_params[k] = agg_params[k] / sum_weight
    #         elif op == "avg":
    #             for k in agg_params.keys():
    #                 agg_params[k] = agg_params[k] / agg_weight
    #         elif op == "sum":
    #             pass

    #     return agg_params








def torch_aggregator(args, raw_grad_list, training_num, op=None):

    if op is None:
        if hasattr(args, "agg_operator"):
            op = args.agg_operator
        else:
            op = "weighted_avg"
    else:
        op = op

    if op == "weighted_avg":
        (num0, avg_params) = raw_grad_list[0]
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k].to(avg_params[k].device) * w
    elif op == "avg":
        (num0, avg_params) = raw_grad_list[0]
        w = 1 / len(raw_grad_list)
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k].to(avg_params[k].device) * w
    elif op == "sum":
        (num0, avg_params) = raw_grad_list[0]
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                if i == 0:
                    avg_params[k] = local_model_params[k]
                else:
                    avg_params[k] += local_model_params[k].to(avg_params[k].device)
    return avg_params


def tf_aggregator(args, raw_grad_list, training_num, op=None):
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


def jax_aggregator(args, raw_grad_list, training_num, op=None):
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


def mxnet_aggregator(args, raw_grad_list, training_num, op=None):
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


def model_aggregator(args, raw_grad_list, training_num, op=None):
    if hasattr(args, MLEngineBackend.ml_engine_args_flag):
        if args.ml_engine == MLEngineBackend.ml_engine_backend_tf:
            return tf_aggregator(args, raw_grad_list, training_num)
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_jax:
            return jax_aggregator(args, raw_grad_list, training_num)
        elif args.ml_engine == MLEngineBackend.ml_engine_backend_mxnet:
            return mxnet_aggregator(args, raw_grad_list, training_num)
        else:
            return torch_aggregator(args, raw_grad_list, training_num, op)
    else:
        return torch_aggregator(args, raw_grad_list, training_num, op)
