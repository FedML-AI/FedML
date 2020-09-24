import logging

from fedml_api.distributed.multitask.message_define import MyMessage
from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message


class DecentralizedWorkerManager(ClientManager):
    def __init__(self, args, comm, rank, size, trainer, topology_manager):
        super().__init__(args, comm, rank, size)
        self.worker_index = rank
        self.trainer = trainer
        self.topology_manager = topology_manager
        self.num_rounds = args.comm_round
        self.round_idx = 0

    def run(self):
        self.start_training()
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR,
                                              self.handle_msg_from_neighbor)
        self.register_message_receive_handler(MyMessage.MSG_TYPE_METRICS,
                                              self.handle_msg_local_test_result)

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_msg_from_neighbor(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)

        self.trainer.add_neighbor_local_result(sender_id, model_params, local_sample_number)
        b_all_received = self.trainer.check_whether_all_receive()
        # logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            self.trainer.aggregate()
            train_acc, train_loss, test_acc, test_loss = self.trainer.test_on_local_data(self.round_idx)
            if train_acc is not None:
                self.send_test_result_worker0(train_acc, train_loss, test_acc, test_loss)

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.num_rounds:
                self.finish()
                return
            self.__train()

    def handle_msg_local_test_result(self, msg_params):
        # logging.info("handle_msg_local_test_result.")
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        train_acc = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_TRAIN_ACC)
        train_loss = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_TRAIN_LOSS)
        test_acc = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_TEST_ACC)
        test_loss = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_TEST_LOSS)
        self.trainer.record_average_test_result(sender_id, self.round_idx, train_acc, train_loss, test_acc, test_loss)

    def __train(self):
        weights, local_sample_num = self.trainer.train(self.round_idx)

        for neighbor_idx in self.topology_manager.get_out_neighbor_idx_list(self.worker_index):
            self.send_result_to_neighbors(neighbor_idx, weights, local_sample_num)

    def send_result_to_neighbors(self, receive_id, weights, local_sample_num):
        # logging.info("send_result_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        message.add_params(MyMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        self.send_message(message)

    def send_test_result_worker0(self, train_acc, train_loss, test_acc, test_loss):
        # logging.info("send_test_result_worker0. receive_id = " + str(0))
        message = Message(MyMessage.MSG_TYPE_METRICS, self.get_sender_id(), 0)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_TRAIN_ACC, train_acc)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_TRAIN_LOSS, train_loss)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_TEST_ACC, test_acc)
        message.add_params(MyMessage.MSG_ARG_KEY_LOCAL_TEST_LOSS, test_loss)
        self.send_message(message)
