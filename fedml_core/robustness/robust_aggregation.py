import torch


def vectorize_weight(state_dict):
    weight_list = []
    for (k, v) in state_dict.items():
        if is_weight_param(k):
            weight_list.append(v)
    return torch.cat(weight_list)


def load_model_weight_diff(local_state_dict, weight_diff, global_state_dict):
    """
    load rule: w_t + clipped(w^{local}_t - w_t)
    """
    recons_local_state_dict = {}
    index_bias = 0
    for item_index, (k, v) in enumerate(local_state_dict.state_dict().items()):
        if is_weight_param(k):
            recons_local_state_dict[k] = weight_diff[index_bias:index_bias + v.numel()].view(v.size()) + \
                                         global_state_dict[k]
            index_bias += v.numel()
        else:
            recons_local_state_dict[k] = v
    return recons_local_state_dict


def is_weight_param(k):
    return ("running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k)


class RobustAggregator(object):
    def __init__(self, args):
        self.defense_type = args.defense_type
        self.norm_bound = args.norm_bound  # for norm diff clipping and weak DP defenses
        self.stddev = args.stddev  # for weak DP defenses

    def norm_diff_clipping(self, local_state_dict, global_state_dict):
        vec_local_weight = vectorize_weight(local_state_dict)
        vec_global_weight = vectorize_weight(global_state_dict)

        # clip the norm diff
        vec_diff = vec_local_weight - vec_global_weight
        weight_diff_norm = torch.norm(vec_diff).item()
        clipped_weight_diff = vec_diff / max(1, weight_diff_norm / self.norm_bound)
        clipped_local_state_dict = load_model_weight_diff(local_state_dict,
                                                          clipped_weight_diff,
                                                          global_state_dict)
        return clipped_local_state_dict

    def add_noise(self, local_weight, device):
        gaussian_noise = torch.randn(local_weight.size(),
                                     device=device) * self.stddev
        dp_weight = local_weight + gaussian_noise
        return dp_weight
