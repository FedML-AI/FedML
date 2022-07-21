from fedml.core.security.common.utils import get_total_sample_num


class AggregationFunction:
    @staticmethod
    def FedAVG(client_grad_list):
        (num0, avg_params) = client_grad_list[0]
        training_num = get_total_sample_num(client_grad_list)
        for k in avg_params.keys():
            for i in range(0, len(client_grad_list)):
                local_sample_number, local_model_params = client_grad_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    avg_params[k] = local_model_params[k] * w
                else:
                    avg_params[k] += local_model_params[k] * w
        return avg_params