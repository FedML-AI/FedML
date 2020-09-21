import logging

import torch

from fedml_api.distributed.fednas.message_define import MyMessage
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.server.server_manager import ServerManager


class FedNASServerManager(ServerManager):
    def __init__(self, args, comm, rank, size, aggregator):
        super().__init__(args, comm, rank, size)

        self.round_num = args.comm_round
        self.round_idx = 0

        self.aggregator = aggregator

    def run(self):
        global_model = self.aggregator.get_model()
        global_model_params = global_model.state_dict()
        global_arch_params = global_model.arch_parameters()
        for process_id in range(1, self.size):
            self.__send_initial_config_to_client(process_id, global_model_params, global_arch_params)
        super().run()

    def register_message_receive_handlers(self):
        self.register_message_receive_handler(MyMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER,
                                              self.__handle_msg_server_receive_model_from_client_opt_send)

    def __send_initial_config_to_client(self, process_id, global_model_params, global_arch_params):
        message = Message(MyMessage.MSG_TYPE_S2C_INIT_CONFIG, self.get_sender_id(), process_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_ARCH_PARAMS, global_arch_params)
        logging.info("MSG_TYPE_S2C_INIT_CONFIG. receiver: " + str(process_id))
        self.send_message(message)

    def __handle_msg_server_receive_model_from_client_opt_send(self, msg_params):
        process_id = msg_params.get(MyMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MyMessage.MSG_ARG_KEY_MODEL_PARAMS)
        arch_params = msg_params.get(MyMessage.MSG_ARG_KEY_ARCH_PARAMS)
        local_sample_number = msg_params.get(MyMessage.MSG_ARG_KEY_NUM_SAMPLES)
        train_acc = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_TRAINING_ACC)
        train_loss = msg_params.get(MyMessage.MSG_ARG_KEY_LOCAL_TRAINING_LOSS)

        self.aggregator.add_local_trained_result(process_id - 1, model_params, arch_params, local_sample_number,
                                                 train_acc, train_loss)
        b_all_received = self.aggregator.check_whether_all_receive()
        logging.info("b_all_received = " + str(b_all_received))
        if b_all_received:
            if self.args.stage == "search":
                global_model_params, global_arch_params = self.aggregator.aggregate()
            else:
                global_model_params = self.aggregator.aggregate()
                global_arch_params = []
            self.aggregator.infer(self.round_idx)  # for NAS, it cost 151 seconds
            self.aggregator.statistics(self.round_idx)
            if self.args.stage == "search":
                self.aggregator.record_model_global_architecture(self.round_idx)

            # free all teh GPU memory cache
            torch.cuda.empty_cache()

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                self.finish()
                return

            for process_id in range(1, self.size):
                self.__send_model_to_client_message(process_id, global_model_params, global_arch_params)

    def __send_model_to_client_message(self, process_id, global_model_params, global_arch_params):
        message = Message(MyMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT, 0, process_id)
        message.add_params(MyMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        message.add_params(MyMessage.MSG_ARG_KEY_ARCH_PARAMS, global_arch_params)
        logging.info("__send_model_to_client_message. MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT. receiver: " + str(process_id))
        self.send_message(message)
