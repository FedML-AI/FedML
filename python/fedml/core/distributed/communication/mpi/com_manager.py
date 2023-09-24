import logging
import queue
import time
from typing import List

from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent
from .mpi_receive_thread import MPIReceiveThread
from ..base_com_manager import BaseCommunicationManager
from ..constants import CommunicationConstants
from ..message import Message
from ..observer import Observer


class MpiCommunicationManager(BaseCommunicationManager):
    """
    MPI Communication Manager.

    This class manages communication using MPI (Message Passing Interface) for federated learning.
    """

    def __init__(self, comm, rank, size):
        """
        Initialize the MPI Communication Manager.

        Args:
            comm: The MPI communicator.
            rank: The rank of the current process.
            size: The total number of processes in the communicator.
        """
        self.comm = comm
        self.rank = rank
        self.size = size

        self._observers: List[Observer] = []

        node_type = "server" if self.rank == 0 else "client"
        if node_type == "client":
            self.q_sender, self.q_receiver = self.init_client_communication()
        elif node_type == "server":
            self.q_sender, self.q_receiver = self.init_server_communication()

        self.server_send_thread = None
        self.server_receive_thread = None
        self.server_collective_thread = None

        # self.client_send_thread = None
        self.client_receive_thread = None
        self.client_collective_thread = None

        self.is_running = True

        time.sleep(5)
        # assert False

    def init_server_communication(self):
        """
        Initialize server-side communication components.

        Returns:
            Tuple: A tuple containing server send and receive queues.
        """
        server_send_queue = queue.Queue(0)
        # self.server_send_thread = MPISendThread(
        #     self.comm, self.rank, self.size, "ServerSendThread", server_send_queue
        # )
        # self.server_send_thread.start()

        server_receive_queue = queue.Queue(0)
        self.server_receive_thread = MPIReceiveThread(
            self.comm, self.rank, self.size, "ServerReceiveThread", server_receive_queue
        )
        self.server_receive_thread.start()

        return server_send_queue, server_receive_queue

    def init_client_communication(self):
        """
        Initialize client-side communication components.

        Returns:
            Tuple: A tuple containing client send and receive queues.
        """
        # SEND
        client_send_queue = queue.Queue(0)
        # self.client_send_thread = MPISendThread(
        #     self.comm, self.rank, self.size, "ClientSendThread", client_send_queue
        # )
        # self.client_send_thread.start()

        # RECEIVE
        client_receive_queue = queue.Queue(0)
        self.client_receive_thread = MPIReceiveThread(
            self.comm, self.rank, self.size, "ClientReceiveThread", client_receive_queue
        )
        self.client_receive_thread.start()

        return client_send_queue, client_receive_queue

    # Ugly delete comments
    # def send_message(self, msg: Message):
    #     self.q_sender.put(msg)

    def send_message(self, msg: Message):
        """
        Send a message using MPI.

        Args:
            msg: The message to be sent.
        """
        # self.q_sender.put(msg)
        dest_id = msg.get(Message.MSG_ARG_KEY_RECEIVER)
        tick = time.time()
        self.comm.send(msg, dest=dest_id)
        MLOpsProfilerEvent.log_to_wandb(
            {"Comm/send_delay": time.time() - tick})

    def add_observer(self, observer: Observer):
        """
        Add an observer to the list of observers.

        Args:
            observer: The observer to be added.
        """
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        """
        Remove an observer from the list of observers.

        Args:
            observer: The observer to be removed.
        """
        self._observers.remove(observer)

    def handle_receive_message(self):
        """
        Handle receiving messages using MPI.

        This function continuously listens for incoming messages and notifies observers when a message is received.
        """
        self.is_running = True
        # the first message after connection, aligned the protocol with MQTT + S3
        self._notify_connection_ready()
        start_listening_time = time.time()
        MLOpsProfilerEvent.log_to_wandb({"ListenStart": start_listening_time})
        while self.is_running:
            if self.q_receiver.qsize() > 0:
                message_handler_start_time = time.time()
                msg_params = self.q_receiver.get()
                self.notify(msg_params)
                MLOpsProfilerEvent.log_to_wandb(
                    {"BusyTime": time.time() - message_handler_start_time}
                )
            time.sleep(0.0001)
        MLOpsProfilerEvent.log_to_wandb(
            {"TotalTime": time.time() - start_listening_time}
        )
        logging.info("!!!!!!handle_receive_message stopped!!!")

    def stop_receive_message(self):
        """
        Stop receiving messages and threads.
        """
        self.is_running = False
        # self.__stop_thread(self.server_send_thread)
        self.__stop_thread(self.server_receive_thread)
        self.__stop_thread(self.server_collective_thread)
        # self.__stop_thread(self.client_send_thread)
        self.__stop_thread(self.client_receive_thread)
        self.__stop_thread(self.client_collective_thread)

    def notify(self, msg_params):
        """
        Notify observers with the received message.

        Args:
            msg_params: The received message.
        """
        msg_type = msg_params.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def _notify_connection_ready(self):
        """
        Notify observers that the connection is ready.
        """
        msg_params = Message()
        msg_params.sender_id = self.rank
        msg_params.receiver_id = self.rank
        msg_type = CommunicationConstants.MSG_TYPE_CONNECTION_IS_READY
        for observer in self._observers:
            try:
                observer.receive_message(msg_type, msg_params)
            except Exception as e:
                logging.warn("Cannot handle connection ready")

    def __stop_thread(self, thread):
        if thread:
            thread.raise_exception()
            thread.join()
