import logging
import time

from fedml_api.distributed.fedavg.message_define import MyMessage
from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication import Message


class FedAVGClientManager(ClientManager):
    def __init__(self, args, comm, rank, size, trainer):
        super().__init__(args, comm, rank, size)
        self.trainer = trainer
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_INIT_CONFIG,
                                              self.handle_message_init)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT,
                                              self.handle_message_receive_model_from_server)

    def handle_message_init(self, msg_params):
        global_model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        self.trainer.update_model(global_model_params)
        self.round_idx = 0
        self.__train()

    def handle_message_receive_model_from_server(self, msg_params):
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        self.trainer.update_model(model_params)
        self.round_idx += 1
        self.__train()
        if self.round_idx == self.num_rounds - 1:
            self.finish()

    def send_model_to_server(self, receive_id, weights, local_sample_num, train_acc, train_loss):
        message = Message(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_TRAINING_ACC, train_acc)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_TRAINING_LOSS, train_loss)
        self.send_message(message)

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        start_time = time.time()
        weights, local_sample_num = self.trainer.train()
        train_finished_time = time.time()
        # for one epoch, the local training time cost is: 75s (based on RTX2080Ti)
        logging.info("local training time cost: %d" % (train_finished_time - start_time))

        train_acc, train_loss = self.trainer.infer()
        infer_finished_time = time.time()
        # for one epoch, the local infer time cost is: 15s (based on RTX2080Ti)
        logging.info("local infer time cost: %d" % (infer_finished_time - train_finished_time))

        self.send_model_to_server(0, weights, local_sample_num, train_acc, train_loss)
        communication_finished_time = time.time()
        # for one epoch, the local communication time cost is: < 1s (based o n RTX2080Ti)
        logging.info("local communication time cost: %d" % (communication_finished_time - infer_finished_time))
