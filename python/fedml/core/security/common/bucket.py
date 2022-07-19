import math


class Bucket:
    @classmethod
    def bucketization(cls, client_grad_list, batch_size):
        (num0, averaged_params) = client_grad_list[0]
        batch_grad_list = []
        for batch_idx in range(0, math.ceil(len(client_grad_list) / batch_size)):
            client_num = cls._get_client_num_current_batch(
                batch_size, batch_idx, client_grad_list
            )
            sample_num = cls._get_total_sample_num_for_current_batch(
                batch_idx * batch_size, client_num, client_grad_list
            )
            batch_weight = dict()
            for i in range(0, client_num):
                local_sample_num, local_model_params = client_grad_list[
                    batch_idx * batch_size + i
                ]
                w = local_sample_num / sample_num
                for k in averaged_params.keys():
                    if i == 0:
                        batch_weight[k] = local_model_params[k] * w
                    else:
                        batch_weight[k] += local_model_params[k] * w
            batch_grad_list.append((sample_num, batch_weight))
        return batch_grad_list

    @staticmethod
    def _get_client_num_current_batch(batch_size, batch_idx, client_grad_list):
        current_batch_size = batch_size
        # not divisible
        if (
            len(client_grad_list) % batch_size > 0
            and batch_idx == math.ceil(len(client_grad_list) / batch_size) - 1
        ):
            current_batch_size = len(client_grad_list) - (batch_idx * batch_size)
        return current_batch_size

    @staticmethod
    def _get_total_sample_num_for_current_batch(
        start, current_batch_size, client_grad_list
    ):
        training_num_for_batch = 0
        for i in range(0, current_batch_size):
            local_sample_number, local_model_params = client_grad_list[start + i]
            training_num_for_batch += local_sample_number
        return training_num_for_batch
