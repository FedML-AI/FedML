import logging
import time

import numpy as np
import wandb

from .utils import transform_list_to_tensor, Saver, EvaluationMetricsKeeper


class FedSegAggregator(object):
    def __init__(self, worker_num, device, model, args, model_trainer):
        self.trainer = model_trainer
        self.worker_num = worker_num
        self.device = device
        self.args = args
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()

        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

        self.train_acc_client_dict = dict()
        self.train_acc_class_client_dict = dict()
        self.train_mIoU_client_dict = dict()
        self.train_FWIoU_client_dict = dict()
        self.train_loss_client_dict = dict()

        self.test_acc_client_dict = dict()
        self.test_acc_class_client_dict = dict()
        self.test_mIoU_client_dict = dict()
        self.test_FWIoU_client_dict = dict()
        self.test_loss_client_dict = dict()

        self.best_mIoU = 0.0
        self.best_mIoU_clients = dict()

        self.saver = Saver(args)
        self.saver.save_experiment_config()

        logging.info(
            "Initializing FedSegAggregator with workers: {0}".format(worker_num)
        )

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("Add model index: {}".format(index))
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info(
            "Aggregating...... {0}, {1}".format(len(self.model_dict), len(model_list))
        )

        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("Aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total), num_clients, replace=False
            )
        logging.info("client_indexes: {}".format(client_indexes))
        return client_indexes

    def add_client_test_result(
        self,
        round_idx,
        client_idx,
        train_eval_metrics: EvaluationMetricsKeeper,
        test_eval_metrics: EvaluationMetricsKeeper,
    ):
        logging.info("Adding client test result : {}".format(client_idx))

        # Populating Training Dictionary
        if round_idx and round_idx % self.args.evaluation_frequency == 0:
            self.train_acc_client_dict[client_idx] = train_eval_metrics.acc
            self.train_acc_class_client_dict[client_idx] = train_eval_metrics.acc_class
            self.train_mIoU_client_dict[client_idx] = train_eval_metrics.mIoU
            self.train_FWIoU_client_dict[client_idx] = train_eval_metrics.FWIoU
            self.train_loss_client_dict[client_idx] = train_eval_metrics.loss

        # Populating Testing Dictionary
        self.test_acc_client_dict[client_idx] = test_eval_metrics.acc
        self.test_acc_class_client_dict[client_idx] = test_eval_metrics.acc_class
        self.test_mIoU_client_dict[client_idx] = test_eval_metrics.mIoU
        self.test_FWIoU_client_dict[client_idx] = test_eval_metrics.FWIoU
        self.test_loss_client_dict[client_idx] = test_eval_metrics.loss

        if self.args.save_client_model:
            best_mIoU = self.best_mIoU_clients.setdefault(client_idx, 0.0)
            test_mIoU = self.test_mIoU_client_dict[client_idx]

            if test_mIoU > best_mIoU:
                self.best_mIoU_clients[client_idx] = test_mIoU
                logging.info(
                    "Saving Model Checkpoint for Client: {0} --> Previous mIoU:{1}; Improved mIoU:{2}".format(
                        client_idx, best_mIoU, test_mIoU
                    )
                )
                is_best = False
                filename = "client" + str(client_idx) + "_checkpoint.pth.tar"
                saver_state = {
                    "best_pred": test_mIoU,
                    "round": round_idx + 1,
                    "state_dict": self.model_dict[client_idx],
                }

                test_eval_metrics_dict = {
                    "accuracy": self.test_acc_client_dict[client_idx],
                    "accuracy_class": self.test_acc_class_client_dict[client_idx],
                    "mIoU": self.test_mIoU_client_dict[client_idx],
                    "FWIoU": self.test_FWIoU_client_dict[client_idx],
                    "loss": self.test_loss_client_dict[client_idx],
                }

                saver_state["test_data_evaluation_metrics"] = test_eval_metrics_dict

                if round_idx and round_idx % self.args.evaluation_frequency == 0:
                    train_eval_metrics_dict = {
                        "accuracy": self.train_acc_client_dict[client_idx],
                        "accuracy_class": self.train_acc_class_client_dict[client_idx],
                        "mIoU": self.train_mIoU_client_dict[client_idx],
                        "FWIoU": self.train_FWIoU_client_dict[client_idx],
                        "loss": self.train_loss_client_dict[client_idx],
                    }
                    saver_state[
                        "train_data_evaluation_metrics"
                    ] = train_eval_metrics_dict

                self.saver.save_checkpoint(saver_state, is_best, filename)

    def output_global_acc_and_loss(self, round_idx):
        logging.info(
            "################## Output global accuracy and loss for round {} :".format(
                round_idx
            )
        )

        if round_idx and round_idx % self.args.evaluation_frequency == 0:
            # Test on training set
            train_acc = np.array(
                [
                    self.train_acc_client_dict[k]
                    for k in self.train_acc_client_dict.keys()
                ]
            ).mean()
            train_acc_class = np.array(
                [
                    self.train_acc_class_client_dict[k]
                    for k in self.train_acc_class_client_dict.keys()
                ]
            ).mean()
            train_mIoU = np.array(
                [
                    self.train_mIoU_client_dict[k]
                    for k in self.train_mIoU_client_dict.keys()
                ]
            ).mean()
            train_FWIoU = np.array(
                [
                    self.train_FWIoU_client_dict[k]
                    for k in self.train_FWIoU_client_dict.keys()
                ]
            ).mean()
            train_loss = np.array(
                [
                    self.train_loss_client_dict[k]
                    for k in self.train_loss_client_dict.keys()
                ]
            ).mean()

            # Train Logs
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Acc_class": train_acc_class, "round": round_idx})
            wandb.log({"Train/mIoU": train_mIoU, "round": round_idx})
            wandb.log({"Train/FWIoU": train_FWIoU, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
            stats = {
                "training_acc": train_acc,
                "training_acc_class": train_acc_class,
                "training_mIoU": train_mIoU,
                "training_FWIoU": train_FWIoU,
                "training_loss": train_loss,
            }
            logging.info("Testing statistics: {}".format(stats))

        # Test on testing set
        test_acc = np.array(
            [self.test_acc_client_dict[k] for k in self.test_acc_client_dict.keys()]
        ).mean()
        test_acc_class = np.array(
            [
                self.test_acc_class_client_dict[k]
                for k in self.test_acc_class_client_dict.keys()
            ]
        ).mean()
        test_mIoU = np.array(
            [self.test_mIoU_client_dict[k] for k in self.test_mIoU_client_dict.keys()]
        ).mean()
        test_FWIoU = np.array(
            [self.test_FWIoU_client_dict[k] for k in self.test_FWIoU_client_dict.keys()]
        ).mean()
        test_loss = np.array(
            [self.test_loss_client_dict[k] for k in self.test_loss_client_dict.keys()]
        ).mean()

        # Test Logs
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Acc_class": test_acc_class, "round": round_idx})
        wandb.log({"Test/mIoU": test_mIoU, "round": round_idx})
        wandb.log({"Test/FWIoU": test_FWIoU, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        stats = {
            "testing_acc": test_acc,
            "testing_acc_class": test_acc_class,
            "testing_mIoU": test_mIoU,
            "testing_FWIoU": test_FWIoU,
            "testing_loss": test_loss,
        }

        logging.info("Testing statistics: {}".format(stats))

        if test_mIoU > self.best_mIoU:
            self.best_mIoU = test_mIoU
            wandb.run.summary["best_mIoU"] = self.best_mIoU
            wandb.run.summary["Round Number for best mIou"] = round_idx
            if self.args.save_model:
                logging.info(
                    "Saving Model Checkpoint --> Previous mIoU:{0}; Improved mIoU:{1}".format(
                        self.best_mIoU, test_mIoU
                    )
                )
                is_best = True

                saver_state = {
                    "best_pred": self.best_mIoU,
                    "round": round_idx + 1,
                    "state_dict": self.trainer.get_model_params(),
                }

                test_eval_metrics_dict = {
                    "accuracy": test_acc,
                    "accuracy_class": test_acc_class,
                    "mIoU": test_mIoU,
                    "FWIoU": test_FWIoU,
                    "loss": test_loss,
                }
                saver_state["test_data_evaluation_metrics"] = test_eval_metrics_dict

                if round_idx and round_idx % self.args.evaluation_frequency == 0:
                    train_eval_metrics_dict = {
                        "accuracy": train_acc,
                        "accuracy_class": train_acc_class,
                        "mIoU": train_mIoU,
                        "FWIoU": train_FWIoU,
                        "loss": train_loss,
                    }
                    saver_state[
                        "train_data_evaluation_metrics"
                    ] = train_eval_metrics_dict

                self.saver.save_checkpoint(saver_state, is_best)
