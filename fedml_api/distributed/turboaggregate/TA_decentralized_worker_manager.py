import logging

from fedml_api.distributed.turboaggregate.message_define import MyMessage
from fedml_core.distributed.client.client_manager import ClientManager
from fedml_core.distributed.communication.message import Message


class TA_DecentralizedWorkerManager(ClientManager):
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

    def start_training(self):
        self.round_idx = 0
        self.__train()

    def handle_msg_from_neighbor(self, msg_params):
        sender_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        training_interation_result = msg_params.get(MyMessage.MSG_ARG_KEY_PARAMS_1)
        logging.info("handle_msg_from_neighbor. sender_id = " + str(sender_id))
        self.trainer.add_result(sender_id, training_interation_result)
        if self.trainer.check_whether_all_receive():
            logging.info(">>>>>>>>>>>>>>>WORKER %d, ROUND %d finished!<<<<<<<<" % (self.worker_index, self.round_idx))
            self.round_idx += 1
            if self.round_idx == self.num_rounds:
                self.finish()
            self.__train()

    def __train(self):
        # do something here (e.g., training)
        training_interation_result = self.trainer.train()

        for neighbor_idx in self.topology_manager.get_out_neighbor_idx_list(self.worker_index):
            self.send_result_to_neighbors(neighbor_idx, training_interation_result)

    def send_message_init_config(self, receive_id):
        message = Message(MyMessage.MSG_TYPE_INIT, self.get_sender_id(), receive_id)
        self.send_message(message)

    def send_result_to_neighbors(self, receive_id, client_params1):
        logging.info("send_result_to_neighbors. receive_id = " + str(receive_id))
        message = Message(MyMessage.MSG_TYPE_SEND_MSG_TO_NEIGHBOR, self.get_sender_id(), receive_id)
        message.add_params(MyMessage.MSG_ARG_KEY_PARAMS_1, client_params1)
        self.send_message(message)
