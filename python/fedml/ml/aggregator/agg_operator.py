from typing import List, Tuple, Dict
from ..engine import ml_engine_adapter


class FedMLAggOperator:
    @staticmethod
    def FedAVG(args, raw_grad_list):
        training_num = 0
        for i in range(len(raw_grad_list)):
            local_sample_num, local_model_params = raw_grad_list[i]
            training_num += local_sample_num

        avg_params = ml_engine_adapter.model_aggregator(args, raw_grad_list, training_num)
        return avg_params

    @staticmethod
    def FedAvg_seq(args, raw_grad_list):
        (num0, avg_params) = raw_grad_list[0]
        for k in avg_params.keys():
            for i in range(0, len(raw_grad_list)):
                local_sample_number, local_model_params = raw_grad_list[i]
                if i == 0:
                    avg_params[k] = local_model_params[k]
                else:
                    avg_params[k] += local_model_params[k]
        return avg_params

    @staticmethod
    def FedOpt(args, raw_grad_list):
        pass

    @staticmethod
    def agg(args, raw_grad_list: List[Tuple[float, Dict]]) -> Dict:
        if args.federated_optimizer == "FedAvg":
            agg_func = FedMLAggOperator.FedAVG
        elif args.federated_optimizer == "FedAvg_seq":
            agg_func = FedMLAggOperator.FedAvg_seq
        elif args.federated_optimizer == "FedOpt":
            agg_func = FedMLAggOperator.FedOpt
        else:
            raise Exception("will support many optimizers in a unified framework soon")
        return agg_func(args, raw_grad_list)




