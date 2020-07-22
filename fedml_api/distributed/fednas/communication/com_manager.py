import logging
import queue
import time
from typing import List

from mpi4py import MPI

from communication.mpi_message import MPIMessage
from communication.mpi_receive_thread import MPIReceiveThread
from communication.mpi_send_thread import MPISendThread
from communication.observer import Observer


class CommunicationManager(object):
    def __init__(self, comm, rank, size, node_type="client"):
        self.comm = comm
        self.rank = rank
        self.size = size

        self._observers: List[Observer] = []

        if node_type == "client":
            self.q_sender, self.q_receiver = self.init_client_communication()
        elif node_type == "server":
            self.q_sender, self.q_receiver = self.init_server_communication()

        self.server_send_thread = None
        self.server_receive_thread = None
        self.server_collective_thread = None

        self.client_send_thread = None
        self.client_receive_thread = None
        self.client_collective_thread = None

        self.is_running = True

    def init_server_communication(self):
        server_send_queue = queue.Queue(0)
        self.server_send_thread = MPISendThread(self.comm, self.rank, self.size, "ServerSendThread", server_send_queue)
        self.server_send_thread.start()

        server_receive_queue = queue.Queue(0)
        self.server_receive_thread = MPIReceiveThread(self.comm, self.rank, self.size, "ServerReceiveThread",
                                                      server_receive_queue)
        self.server_receive_thread.start()

        return server_send_queue, server_receive_queue

    def init_client_communication(self):
        # SEND
        client_send_queue = queue.Queue(0)
        self.client_send_thread = MPISendThread(self.comm, self.rank, self.size, "ClientSendThread", client_send_queue)
        self.client_send_thread.start()

        # RECEIVE
        client_receive_queue = queue.Queue(0)
        self.client_receive_thread = MPIReceiveThread(self.comm, self.rank, self.size, "ClientReceiveThread",
                                                      client_receive_queue)
        self.client_receive_thread.start()

        return client_send_queue, client_receive_queue

    def send_message(self, msg: MPIMessage):
        self.q_sender.put(msg)

    def send_broadcast_collective_message(self, msg: MPIMessage):
        operation = msg.get(MPIMessage.MSG_ARG_KEY_OPERATION)
        if operation == MPIMessage.MSG_OPERATION_BROADCAST:
            self.comm.bcast(msg.to_string(), root=0)

    def receive_broadcast_collective_message(self):
        msg_str = None
        msg_str = self.comm.bcast(msg_str, root=0)
        if not msg_str:
            return None
        msg = MPIMessage()
        msg.init(msg_str)
        return msg

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        self.is_running = True
        while self.is_running:
            if self.q_receiver.qsize() > 0:
                msg_params = self.q_receiver.get()
                self.notify(msg_params)

            msg_params = self.receive_broadcast_collective_message()
            if msg_params:
                self.notify(msg_params)

            time.sleep(0.3)
        logging.info("!!!!!!handle_receive_message stopped!!!")

    def stop_receive_message(self):
        self.is_running = False
        self.__stop_thread(self.server_send_thread)
        self.__stop_thread(self.server_receive_thread)
        self.__stop_thread(self.server_collective_thread)
        self.__stop_thread(self.client_send_thread)
        self.__stop_thread(self.client_receive_thread)
        self.__stop_thread(self.client_collective_thread)

    def notify(self, msg_params):
        msg_type = msg_params.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def __stop_thread(self, thread):
        if thread:
            thread.raise_exception()
            thread.join()
