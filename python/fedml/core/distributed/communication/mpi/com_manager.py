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
    
    def __init__(self, comm, rank, size):
        self.comm = comm
        self.rank = rank
        self.size = size

        self._observers: List[Observer] = []

        self._stop_running = False
        self._server_receive_thread = None # Server-Thread
        self._client_receive_thread = None # Client-Thread
        self.q_sender = None # Queue-1
        self.q_receiver = None # Queue-2

        node_type = "server" if self.rank == 0 else "client"
        if node_type == "client":
            self.q_sender, self.q_receiver = self.init_client_communication()
        elif node_type == "server":
            self.q_sender, self.q_receiver = self.init_server_communication()

        time.sleep(0.001)

    def init_server_communication(self):
        # SEND
        server_send_queue = queue.Queue(0)

        # RECEIVE
        server_receive_queue = queue.Queue(0)
        self._server_receive_thread = MPIReceiveThread(
            self.comm,
            self.rank,
            self.size,
            "ServerReceiveThread",
            server_receive_queue)
        self._server_receive_thread.start()

        return server_send_queue, server_receive_queue

    def init_client_communication(self):
        # SEND
        client_send_queue = queue.Queue(0)

        # RECEIVE
        client_receive_queue = queue.Queue(0)
        self._client_receive_thread = MPIReceiveThread(
            self.comm, 
            self.rank, 
            self.size, 
            "ClientReceiveThread", 
            client_receive_queue)
        self._client_receive_thread.start()

        return client_send_queue, client_receive_queue

    def send_message(self, msg: Message):
        dest_id = msg.get(Message.MSG_ARG_KEY_RECEIVER)
        tick = time.time()
        self.comm.send(msg, dest=dest_id)
        MLOpsProfilerEvent.log_to_wandb({"Comm/send_delay": time.time() - tick})

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        # the first message after connection, aligned the protocol with MQTT + S3
        self._notify_connection_ready()
        start_listening_time = time.time()
        MLOpsProfilerEvent.log_to_wandb({"ListenStart": start_listening_time})
        while not self._stop_running:
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
        logging.info("Handle receive message stopped!")

    def stop_receive_message(self):
        self._stop_manager()        

    def _stop_manager(self):        
        self._stop_running = True
        self._release_resources()

    def _release_resources(self):
        for t in [self._server_receive_thread, self._client_receive_thread]:
            self._stop_thread(t)
        for q in [self.q_sender, self.q_receiver]:
            self._free_q(q)      

    def _stop_thread(self, mpi_rcv_thread):
        if mpi_rcv_thread and isinstance(mpi_rcv_thread, MPIReceiveThread):
            mpi_rcv_thread.stop()
            mpi_rcv_thread.join()

    def _free_q(self, q):
        while not q.empty():            
            q.get_no_wait()
            q.task_done()            

    def notify(self, msg_params):
        msg_type = msg_params.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)

    def _notify_connection_ready(self):
        msg_params = Message()
        msg_params.sender_id = self.rank
        msg_params.receiver_id = self.rank
        msg_type = CommunicationConstants.MSG_TYPE_CONNECTION_IS_READY
        for observer in self._observers:
            try:
                observer.receive_message(msg_type, msg_params)
            except Exception as e:
                logging.warn("Cannot handle connection ready.")
            