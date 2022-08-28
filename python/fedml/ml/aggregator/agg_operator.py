from typing import List, Tuple, Dict

from ...core.common.ml_engine_backend import MLEngineBackend


class FedMLAggOperator:
    @staticmethod
    def agg(args, raw_grad_list: List[Tuple[float, Dict]]) -> Dict:
        training_num = 0
        for i in range(len(raw_grad_list)):
            local_sample_num, local_model_params = raw_grad_list[i]
            training_num += local_sample_num

        avg_params = model_aggregator(args, raw_grad_list, training_num)
        return avg_params


def torch_aggregator(args, raw_grad_list, training_num):
    (num0, avg_params) = raw_grad_list[0]

    if args.federated_optimizer == "FedAvg":
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k] * w
    elif args.federated_optimizer == "FedAvg_seq":
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                if i == 0:
                    avg_params[k] = local_model_params[k]
                else:
                    avg_params[k] += local_model_params[k]
    elif args.federated_optimizer == "FedOpt":
        pass

    return avg_params


def tf_aggregator(args, raw_grad_list, training_num):
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
