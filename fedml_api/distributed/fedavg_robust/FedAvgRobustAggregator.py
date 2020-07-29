import logging
import time

import torch
import wandb
from torch import nn


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
            recons_local_state_dict[k] = weight_diff[index_bias:index_bias+v.numel()].view(v.size()) + global_state_dict[k]
            index_bias += v.numel()
        else:
            recons_local_state_dict[k] = v
    return recons_local_state_dict


def add_noise(local_weight, stddev, device):
    gaussian_noise = torch.randn(local_weight.size(),
                            device=device) * self.stddev
    dp_weight = vectorized_net + gaussian_noise
    return dp_weight


def is_weight_param(key_name):
    return ("running_mean" not in k and "running_var" not in k and "num_batches_tracked" not in k)


class FedAvgRobustAggregator(object):
    def __init__(self, train_global, test_global, all_train_data_num, client_num, device, model, args):
        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num
        self.client_num = client_num
        self.device = device
        self.args = args
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.train_acc_dict = dict()
        self.train_loss_dict = dict()
        self.test_acc_avg = 0.0
        self.test_loss_avg = 0.0
        self.flag_client_model_uploaded_dict = dict()

        self.defense_type = args.defense_type
        self.norm_bound = args.norm_bound # for norm diff clipping and weak DP defenses
        self.stddev = args.stddev # for weak DP defenses

        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        self.model, _ = self.init_model(model)

    def init_model(self, model):
        model_params = model.state_dict()
        logging.info(model)
        return model, model_params

    def get_global_model_params(self):
        return self.model.state_dict()

    def add_local_trained_result(self, index, model_params, sample_num, train_acc, train_loss):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.train_acc_dict[index] = train_acc
        self.train_loss_dict[index] = train_loss
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.client_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.client_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        
        vec_global_weight = vectorize_state_dict(self.model.state_dict())

        for idx in range(self.client_num):
            local_sample_number, local_model_params = model_list[i]
            # conduct the defense here:
            if self.defense_type in ("norm_diff_clipping", "weak_dp"):
                vec_local_weight = vectorize_state_dict(local_model_params)
                # clip the norm diff
                vec_diff = vec_local_weight - vec_global_weight
                weight_diff_norm = torch.norm(vectorize_diff).item()
                clipped_weight_diff = vectorize_diff/max(1, weight_diff_norm/self.norm_bound)
                clipped_local_state_dict = load_model_weight_diff(local_model_params, 
                                                                  clipped_weight_diff, 
                                                                  self.model.state_dict())
            else:
                raise NotImplementedError("Non-supported Defense type ... ")

            model_list.append((local_sample_number, clipped_local_state_dict))


        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]

        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / self.all_train_data_num

                local_layer_update = local_model_params[k]

                if self.defense_type == "weak_dp":
                    if is_weight_param(k):
                        local_layer_update = add_noise(local_layer_update, self.stddev, self.device)

                if i == 0:
                    averaged_params[k] = local_layer_update * w
                else:
                    averaged_params[k] += local_layer_update * w

        # update the global model which is cached at the server side
        self.model.load_state_dict(averaged_params)

        # clear the memory cost
        model_list.clear()
        del model_list
        self.model_dict.clear()
        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def statistics(self, round_idx):
        # train acc
        train_acc_list = self.train_acc_dict.values()
        train_acc_avg = sum(train_acc_list) / len(train_acc_list)
        logging.info('Round {:3d}, Average Train Accuracy {:.3f}'.format(round_idx, train_acc_avg))
        wandb.log({"Train/AccTop1": train_acc_avg, "round": round_idx + 1})

        # train loss
        train_loss_list = self.train_loss_dict.values()
        train_loss_avg = sum(train_loss_list) / len(train_loss_list)
        logging.info('Round {:3d}, Average Train Loss {:.3f}'.format(round_idx, train_loss_avg))
        wandb.log({"Train/Loss": train_loss_avg, "round": round_idx + 1})

        # algorithms acc
        logging.info('Round {:3d}, Average Validation Accuracy {:.3f}'.format(round_idx, self.test_acc_avg))
        wandb.log({"Test/AccTop1": self.test_acc_avg, "round": round_idx + 1})

        # algorithms loss
        logging.info('Round {:3d}, Average Validation Loss {:.3f}'.format(round_idx, self.test_loss_avg))
        wandb.log({"Test/Loss": self.test_loss_avg, "round": round_idx + 1})

    def infer(self, round_idx):
        if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
            start_time = time.time()
            self.model.eval()
            self.model.to(self.device)

            test_correct = 0.0
            test_loss = 0.0
            test_sample_number = 0.0
            test_data = self.test_global
            # loss
            criterion = nn.CrossEntropyLoss().to(self.device)
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(test_data):
                    x = x.to(self.device)
                    target = target.to(self.device)

                    pred = self.model(x)
                    loss = criterion(pred, target)
                    _, predicted = torch.max(pred, 1)
                    correct = predicted.eq(target).sum()

                    test_correct += correct.item()
                    test_loss += loss.item() * target.size(0)
                    test_sample_number += target.size(0)
                logging.info("server algorithms. round_idx = %d, test_loss = %s" % (round_idx, test_loss))

            self.test_acc_avg = test_correct / test_sample_number
            self.test_acc_avg = self.test_acc_avg
            logging.info("self.test_acc_avg = " + str(self.test_acc_avg))
            self.test_loss_avg = test_loss

            end_time = time.time()
            logging.info("server_infer time cost: %d" % (end_time - start_time))
