import logging
import time

from .message_define import MyMessage
from ....core.distributed.fedml_comm_manager import FedMLCommManager
from ....core.distributed.communication.message import Message


class FedNASClientManager(FedMLCommManager):
    def __init__(self, args, comm, rank, size, trainer):
        super().__init__(args, comm, rank, size)

        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.args.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.__handle_msg_client_receive_config
        )
        self.register_message_receive_handler(
            MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
            self.__handle_msg_client_receive_model_from_server,
        )

    def __handle_msg_client_receive_config(self, msg_params):
        logging.info("__handle_msg_client_receive_config")
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        arch_params = msg_params.get(MyMessage.MSG_ARG_KEY_ARCH_PARAMS)
        self.trainer.update_model(global_model_params)
        if self.args.stage == "search":
            self.trainer.update_arch(arch_params)

        self.args.round_idx = 0
        # start to train
        self.__train()

    def __handle_msg_client_receive_model_from_server(self, msg_params):
        process_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        arch_params = msg_params.get(MyMessage.MSG_ARG_KEY_ARCH_PARAMS)
        if process_id != 0:
            return
        self.trainer.update_model(model_params)
        if self.args.stage == "search":
            self.trainer.update_arch(arch_params)

        self.args.round_idx += 1
        self.__train()
        if self.args.round_idx == self.num_rounds - 1:
            self.finish()

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.args.round_idx)
        start_time = time.time()
        if self.args.stage == "search":
            (
                weights,
                alphas,
                local_sample_num,
                train_acc,
                train_loss,
            ) = self.trainer.search()
        else:
            weights, local_sample_num, train_acc, train_loss = self.trainer.train()
            alphas = []
        train_finished_time = time.time()
        # for one epoch, the local searching time cost is: 75s (based on RTX2080Ti)
        logging.info(
            "local searching time cost: %d" % (train_finished_time - start_time)
        )

        self.__send_msg_fedavg_send_model_to_server(
            weights, alphas, local_sample_num, train_acc, train_loss
        )
        communication_finished_time = time.time()
        # for one epoch, the local communication time cost is: < 1s (based o n RTX2080Ti)
        logging.info(
            "local communication time cost: %d"
            % (communication_finished_time - train_finished_time)
        )

    def __send_msg_fedavg_send_model_to_server(
        self, weights, alphas, local_sample_num, valid_acc, valid_loss
    ):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.rank, 0)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_ARCH_PARAMS, alphas)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_TRAINING_ACC, valid_acc)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_TRAINING_LOSS, valid_loss)
        self.send_message(message)
