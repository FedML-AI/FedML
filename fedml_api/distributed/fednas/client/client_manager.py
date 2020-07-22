import logging
import sys
import time

from communication.com_manager import CommunicationManager
from communication.mpi_message import MPIMessage
from communication.observer import Observer


class ClientMananger(Observer):

    def __init__(self, args, comm, rank, size, round_num, trainer):
        self.args = args
        self.size = size
        self.rank = rank
        self.com_manager = CommunicationManager(comm, rank, size, node_type="client")
        self.com_manager.add_observer(self)

        self.trainer = trainer
        self.num_rounds = round_num
        self.round_idx = 0

    def receive_message(self, msg_type, msg_params) -> None:
        logging.info("receive_message. rank_id = %d, msg_type = %s. msg_params = %s" % (
            self.rank, str(msg_type), str(msg_params.get_content())))
        if msg_type == MPIMessage.MSG_TYPE_S2C_INIT_CONFIG:
            logging.info("MSG_TYPE_S2C_INIT_CONFIG.")
            self.__handle_msg_client_receive_config(msg_params)
        elif msg_type == MPIMessage.MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT:
            logging.info("MSG_TYPE_S2C_SYNC_MODEL_TO_CLIENT.")
            self.__handle_msg_client_receive_model_from_server(msg_params)

    def run(self):
        self.com_manager.handle_receive_message()

    def __handle_msg_client_receive_config(self, msg_params):
        process_id = msg_params.get(MPIMessage.MSG_ARG_KEY_SENDER)
        global_model_params = msg_params.get(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS)
        arch_params = msg_params.get(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS)
        if process_id != 0:
            return
        self.trainer.update_model(global_model_params)
        if self.args.stage == "search":
            self.trainer.update_arch(arch_params)

        self.round_idx = 0
        # start to train
        self.__train()

    def __handle_msg_client_receive_model_from_server(self, msg_params):
        process_id = msg_params.get(MPIMessage.MSG_ARG_KEY_SENDER)
        model_params = msg_params.get(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS)
        arch_params = msg_params.get(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS)
        if process_id != 0:
            return
        self.trainer.update_model(model_params)
        if self.args.stage == "search":
            self.trainer.update_arch(arch_params)

        self.round_idx += 1
        self.__train()
        if self.round_idx == self.num_rounds - 1:
            self.__finish()

    def __train(self):
        logging.info("#######training########### round_id = %d" % self.round_idx)
        start_time = time.time()
        if self.args.stage == "search":
            weights, alphas, local_sample_num, train_acc, train_loss = self.trainer.search()
        else:
            weights, local_sample_num, train_acc, train_loss = self.trainer.train()
            alphas = []
        train_finished_time = time.time()
        # for one epoch, the local searching time cost is: 75s (based on RTX2080Ti)
        logging.info("local searching time cost: %d" % (train_finished_time - start_time))

        """
        In order to maintain the flexibility and track the asynchronous property, we temporarily don't use "comm.reduce".
        The communication speed of CUDA-aware MPI is faster than regular MPI. 
        According to this document:
        http://on-demand.gputechconf.com/gtc/2014/presentations/S4236-multi-gpu-programming-mpi.pdf,
        when the message size is around 4 Megabytes, CUDA-aware MPI is three times faster than regular MPI.
        In our case, the model size of ResNet is around 229.62M. Thus, we will optimize the communication speed using
        CUDA-aware MPI.
        """
        self.__send_msg_fedavg_send_model_to_server(weights, alphas, local_sample_num, train_acc, train_loss)
        communication_finished_time = time.time()
        # for one epoch, the local communication time cost is: < 1s (based o n RTX2080Ti)
        logging.info("local communication time cost: %d" % (communication_finished_time - train_finished_time))
        """
        According to the above metric, we can estimate the total time cost on all workers:
        worker number: 16
        dataset: CIFAR10
        GPU device: RTX2080Ti
        Local Epoch: 5
        Communication Rounds: 100
        Total time cost on the local per round: T = worker_number * ((training_time_one_epoch * local_epochs) + 
                                                    infer_time + communication_time_cost)
        If we do not parallelize the computing using multi-processes and multi-GPUs, the time cost will be:
                            16 * (75*5+15) = 6240s / round
        The entire training time is larger than 6240s * 100 / 3600 = 173.3 hours = 7 days, which is prohibited in the real-world.
        When optimized by using MPI-based distributed computing, the final training time is reduced to:
                        (75*5+15) * 100 / 3600 = 10.8 hours
        Compared to the centralized training which cost around (75*16 + 15)*100 / 3600 = 33.7 hours
        """

    def __send_msg_fedavg_send_model_to_server(self, weights, alphas, local_sample_num, valid_acc, valid_loss):
        msg = MPIMessage()
        msg.add(MPIMessage.MSG_ARG_KEY_OPERATION, MPIMessage.MSG_OPERATION_SEND)
        msg.add(MPIMessage.MSG_ARG_KEY_TYPE, MPIMessage.MSG_TYPE_C2S_SEND_MODEL_TO_SERVER)
        msg.add(MPIMessage.MSG_ARG_KEY_SENDER, self.rank)
        msg.add(MPIMessage.MSG_ARG_KEY_RECEIVER, 0)
        msg.add(MPIMessage.MSG_ARG_KEY_NUM_SAMPLES, local_sample_num)
        msg.add(MPIMessage.MSG_ARG_KEY_MODEL_PARAMS, weights)
        msg.add(MPIMessage.MSG_ARG_KEY_ARCH_PARAMS, alphas)
        msg.add(MPIMessage.MSG_ARG_KEY_LOCAL_TRAINING_ACC, valid_acc)
        msg.add(MPIMessage.MSG_ARG_KEY_LOCAL_TRAINING_LOSS, valid_loss)
        self.com_manager.send_message(msg)

    def __finish(self):
        logging.info("#######finished########### rank = %d" % self.rank)
        self.com_manager.stop_receive_message()
        logging.info("sys.exit(0)")
        sys.exit()
