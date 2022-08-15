from typing import List, Tuple, Dict


class FedMLAggOperator:
    @staticmethod
    def FedAVG(raw_grad_list):
        (num0, avg_params) = raw_grad_list[0]
        training_num = 0
        for i in range(len(raw_grad_list)):
            local_sample_num, local_model_params = raw_grad_list[i]
            training_num += local_sample_num

        if isinstance(avg_params, dict):
            for k in avg_params.keys():
                for i in range(0, len(raw_grad_list)):
                    local_sample_number, local_model_params = raw_grad_list[i]
                    w = local_sample_number / training_num
                    if i == 0:
                        avg_params[k] = local_model_params[k] * w
                    else:
                        avg_params[k] += local_model_params[k] * w
        elif isinstance(avg_params, list):
            for k in range(0, len(avg_params)):
                for i in range(0, len(raw_grad_list)):
                    local_sample_number, local_model_params = raw_grad_list[i]
                    w = local_sample_number / training_num
                    if i == 0:
                        avg_params[k] = local_model_params[k] * w
                    else:
                        avg_params[k] += local_model_params[k] * w
        return avg_params

    @staticmethod
    def FedAvg_seq(raw_grad_list):
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
    def FedOpt(raw_grad_list):
        pass


    @staticmethod
    def agg(args, raw_grad_list: List[Tuple[float, Dict]]) -> Dict:
        if args.federated_optimizer == "FedAvg":
            agg_func = FedMLAggOperator.FedAVG
        elif args.federated_optimizer == "FedAvg_seq":
            agg_func = FedMLAggOperator.FedAVG_seq
        elif args.federated_optimizer == "FedOpt":
            agg_func = FedMLAggOperator.FedOpt
        else:
            raise Exception("will support many optimizers in a unified framework soon")
        return agg_func(raw_grad_list)




