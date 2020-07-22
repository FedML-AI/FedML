import logging
import sys

import torch

from communication.com_manager import CommunicationManager
from communication.mpi_message import MPIMessage
from communication.observer import Observer


class ServerMananger(Observer):

    def __init__(self, args, comm, rank, size, round_num, aggregator):
        self.args = args
        self.size = size
        self.rank = rank
        self.round_num = round_num
        self.round_idx = 0
        self.com_manager = CommunicationManager(comm, rank, size, node_type="server")
        self.com_manager.add_observer(self)

        self.aggregator = aggregator

    def receive_message(self, msg_type, msg_params) -> None:
        logging.info("receive_message. rank_id = %d, msg_type = %s. msg_params = %s" % (
            self.rank, str(msg_type), str(msg_params.get_content())))
        if msg_type == MPIMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER:
            logging.info("MSG_TYPE_C2S_SEND_MODEL_TO_SERVER.")
            self.__handle_msg_server_receive_model_from_client_opt_send(msg_params)

    def run(self):
        self.init_config()
        self.com_manager.handle_receive_message()

    def init_config(self):
        self.__broadcast_initial_config_to_client()
        """
        comm.bcast (tree structure) is faster than a loop send/receive operation:
        https://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/
        """
        # for process_id in range(1, self.size):
        #     self.__send_initial_config_to_client(process_id)

    def __send_initial_config_to_client(self, process_id):
        global_model = self.aggregator.get_model()
        global_model_params = global_model.state_dict()
        global_arch_params = global_model.arch_parameters()
        msg = MPIMessage()
        msg.add(MPIMessage.MSG_ARG_KEY_OPERATION, MPIMessage.MSG_OPERATION_SEND)
        msg.add(MPIMessage.MSG_ARG_KEY_TYPE, MPIMessage.MSG_TYPE_S2C_INIT_CONFIG)
        msg.add(MPIMessage.MSG_ARG_KEY_SENDER, 0)
        msg.add(MPIMessage.MSG_ARG_KEY_RECEIVER, process_id)
        msg.add(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        msg.add(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS, global_arch_params)
        logging.info("MSG_TYPE_S2C_INIT_CONFIG. receiver: " + str(process_id))
        self.com_manager.send_message(msg)

    def __broadcast_initial_config_to_client(self):
        global_model = self.aggregator.get_model()
        global_model_params = global_model.state_dict()
        global_arch_params = []
        if self.args.stage == "search":
            global_arch_params = global_model.arch_parameters()
        msg = MPIMessage()
        msg.add(MPIMessage.MSG_ARG_KEY_OPERATION, MPIMessage.MSG_OPERATION_BROADCAST)
        msg.add(MPIMessage.MSG_ARG_KEY_TYPE, MPIMessage.MSG_TYPE_S2C_INIT_CONFIG)
        msg.add(MPIMessage.MSG_ARG_KEY_SENDER, 0)
        msg.add(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        msg.add(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS, global_arch_params)
        logging.info("__broadcast_initial_config_to_client. MSG_TYPE_S2C_INIT_CONFIG.")
        self.com_manager.send_broadcast_collective_message(msg)

    def __handle_msg_server_receive_model_from_client_opt_send(self, msg_params):
        process_id = msg_params.get(MPIMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS)
        arch_params = msg_params.get(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS)
        local_sample_number = msg_params.get(MPIMessage.MSG_ARG_KEY_NUM_SAMPLES)
        train_acc = msg_params.get(MPIMessage.MSG_ARG_KEY_LOCAL_TRAINING_ACC)
        train_loss = msg_params.get(MPIMessage.MSG_ARG_KEY_LOCAL_TRAINING_LOSS)

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
            self.aggregator.infer(self.round_idx) # for NAS, it cost 151 seconds
            self.aggregator.statistics(self.round_idx)
            if self.args.stage == "search":
                self.aggregator.record_model_global_architecture(self.round_idx)

            # free all teh GPU memory cache
            torch.cuda.empty_cache()

            # start the next round
            self.round_idx += 1
            if self.round_idx == self.round_num:
                self.__finish()
                return

            """
            comm.bcast (tree structure) is faster than a loop send/receive operation:
            https://mpitutorial.com/tutorials/mpi-broadcast-and-collective-communication/
            """
            for process_id in range(1, self.size):
                self.__send_model_to_client_message(process_id, global_model_params, global_arch_params)
            # self.__broadcast_model_to_client_message(global_model_params, global_arch_params)

    def __send_model_to_client_message(self, process_id, global_model_params, global_arch_params):
        msg = MPIMessage()
        msg.add(MPIMessage.MSG_ARG_KEY_OPERATION, MPIMessage.MSG_OPERATION_SEND)
        msg.add(MPIMessage.MSG_ARG_KEY_TYPE, MPIMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT)
        msg.add(MPIMessage.MSG_ARG_KEY_SENDER, 0)
        msg.add(MPIMessage.MSG_ARG_KEY_RECEIVER, process_id)
        msg.add(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        msg.add(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS, global_arch_params)
        logging.info("__send_model_to_client_message. MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT. receiver: " + str(process_id))
        self.com_manager.send_message(msg)

    def __broadcast_model_to_client_message(self, global_model_params, global_arch_params):
        msg = MPIMessage()
        msg.add(MPIMessage.MSG_ARG_KEY_OPERATION, MPIMessage.MSG_OPERATION_BROADCAST)
        msg.add(MPIMessage.MSG_ARG_KEY_TYPE, MPIMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT)
        msg.add(MPIMessage.MSG_ARG_KEY_SENDER, 0)
        msg.add(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS, global_model_params)
        msg.add(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS, global_arch_params)
        logging.info("__broadcast_model_to_client_message. MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT. bcast")
        self.com_manager.send_broadcast_collective_message(msg)

    def __finish(self):
        logging.info("__finish server")
        self.com_manager.stop_receive_message()
        logging.info("sys.exit(0)")
        sys.exit()
