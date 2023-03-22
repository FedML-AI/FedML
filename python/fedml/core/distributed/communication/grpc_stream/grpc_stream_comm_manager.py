import os
import pickle
import threading
from concurrent import futures
from typing import List
import grpc
import sys

from . import grpc_stream_comm_manager_pb2_grpc, grpc_stream_comm_manager_pb2

lock = threading.Lock()

from fedml.core.distributed.communication.base_com_manager import BaseCommunicationManager
from fedml.core.distributed.communication.message import Message
from fedml.core.distributed.communication.observer import Observer
from ..constants import CommunicationConstants

from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent

import time

# Check Service or serve?
from fedml.core.distributed.communication.grpc_stream.grpc_stream_server import GRPCStreamCOMMServicer
from fedml.core.distributed.communication.grpc_stream.queue_manager import QueueManager

import logging

import csv




class GRPCStreamCommManager(BaseCommunicationManager):
    def __init__(
        self,
        host,
        port,
        server_ip,
        topic="fedml",
        client_id=0,
        client_num=0,
    ):
        # host is the ip address of server
        self.host = host
        self.port = str(port)
        self._topic = topic
        self.client_id = client_id
        self.client_num = client_num
        self._observers: List[Observer] = []
        self.rank = client_id
        self.queue_manager = QueueManager()
        self.server_ip = server_ip
        self.opts = [
            ("grpc.max_concurrent_streams", client_num),
            ("grpc.max_send_message_length", 1000 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1000 * 1024 * 1024),
            ("grpc.enable_http_proxy", 0),
        ]
        if client_id == 0:
            self.node_type = "server"
            logging.info("############# THIS IS FL SERVER ################")
            self.grpc_server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=client_num * 2),
                options=self.opts,
            )
            self.grpc_servicer = GRPCStreamCOMMServicer(host, port, client_num, client_id, self.queue_manager)
            grpc_stream_comm_manager_pb2_grpc.add_GRPCStreamCommManagerServicer_to_server(
                self.grpc_servicer, self.grpc_server
            )
            logging.info(os.getcwd())

            # starts a grpc_server on local machine using ip address "0.0.0.0"
            self.grpc_server.add_insecure_port("{}:{}".format("0.0.0.0", port))

            self.grpc_server.start()
            logging.info("grpc server started. Listening on port " + str(port))
        else:
            self.node_type = "client"
            logging.info("------------- THIS IS FL CLIENT ----------------")
            receiver_id = 0
            receiver_ip = self.server_ip
            PORT_BASE = CommunicationConstants.GRPC_BASE_PORT
            channel_url = "{}:{}".format(receiver_ip, str(PORT_BASE + receiver_id))
            channel = grpc.insecure_channel(channel_url, options=self.opts)
            stub = grpc_stream_comm_manager_pb2_grpc.GRPCStreamCommManagerStub(channel)
            request = grpc_stream_comm_manager_pb2.ConnectRequest(client_id = client_id)
            # Connect to server to initiate server to client stream
            self.responses = stub.Connect(request)
            self.is_running = True


    def send_message(self, msg: Message):
        logging.info("msg = {}".format(msg))
        logging.info("pickle.dumps(msg) START")
        pickle_dump_start_time = time.time()
        msg_pkl = pickle.dumps(msg)
        MLOpsProfilerEvent.log_to_wandb({"PickleDumpsTime": time.time() - pickle_dump_start_time})
        logging.info("pickle.dumps(msg) END")
        receiver_id = msg.get_receiver_id()
        request = grpc_stream_comm_manager_pb2.CommRequest()
        request.client_id = self.client_id
        request.message = msg_pkl

        # Send the message based on the server type
        # Using different RPC type (Streaming for s-to-c and Unary for c-to-s) makes
        # it easier to support more complex messaging order
        if self.client_id == 0:
            self.send_from_server(request, receiver_id)
        else:
            self.send_from_client(request, receiver_id)


    def send_from_server(self, request, receiver_id):
        logging.info("sending message to server")
        # Add to queue to be picked up by the stream handler (Connect method of GRPCStreamCOMMServicer)
        self.queue_manager.add_to_send_messages_queue(receiver_id, request)

    def send_from_client(self, request, receiver_id):
        # Send the message using Unary RPC call
        PORT_BASE = CommunicationConstants.GRPC_BASE_PORT
        receiver_ip = self.server_ip
        channel_url = "{}:{}".format(receiver_ip, str(PORT_BASE + receiver_id))
        channel = grpc.insecure_channel(channel_url, options=self.opts)
        stub = grpc_stream_comm_manager_pb2_grpc.GRPCStreamCommManagerStub(channel)
        logging.info("sending message to {}".format(channel_url))
        tick = time.time()
        stub.SendToServer(request)
        MLOpsProfilerEvent.log_to_wandb({"Comm/send_delay": time.time() - tick})
        logging.debug("sent successfully")
        channel.close()


    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        self._notify_connection_ready()
        if self.client_id == 0:
            self.message_handling_subroutine()
        # Cannont run message_handling_subroutine in new thread
        # Related https://stackoverflow.com/a/70705165
        else:        
            thread = threading.Thread(target=self.message_handling_subroutine)
            thread.start()
            try:
                for response in self.responses:
                    self.queue_manager.add_to_received_messages_queue(response.message)
            except grpc._channel._MultiThreadedRendezvous as e:
                if (e.code() == grpc.StatusCode.UNAVAILABLE):
                    logging.info("GRPC Connection Disconnected or Unavailable")
                    print(e)

    def message_handling_subroutine(self):
        start_listening_time = time.time()
        MLOpsProfilerEvent.log_to_wandb({"ListenStart": start_listening_time})
        # Wait for new message in queue
        for msg_pkl in self.queue_manager.get_received_messages_iterator():
            lock.acquire()
            busy_time_start_time = time.time()
            logging.info("unpickle START")
            unpickle_start_time = time.time()
            msg = pickle.loads(msg_pkl)
            MLOpsProfilerEvent.log_to_wandb({"UnpickleTime": time.time() - unpickle_start_time})
            logging.info("unpickle END")
            msg_type = msg.get_type()
            for observer in self._observers:
                _message_handler_start_time = time.time()
                observer.receive_message(msg_type, msg)
                MLOpsProfilerEvent.log_to_wandb({"MessageHandlerTime": time.time() - _message_handler_start_time})
            MLOpsProfilerEvent.log_to_wandb({"BusyTime": time.time() - busy_time_start_time})
            lock.release()
        logging.info("Finished")
        MLOpsProfilerEvent.log_to_wandb({"TotalTime": time.time() - start_listening_time})

    def stop_receive_message(self):
        self.is_running = False
        # Finish waiting for iterators of queue
        self.queue_manager.stop_iterators()
        if self.client_id == 0:
            self.grpc_server.stop(None)
        else:
            # Cancel stream if still alive from server
            self.responses.cancel()


    def notify(self, message: Message):
        msg_type = message.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, message)

    def _notify_connection_ready(self):
        msg_params = Message()
        msg_params.sender_id = self.rank
        msg_params.receiver_id = self.rank
        msg_type = CommunicationConstants.MSG_TYPE_CONNECTION_IS_READY
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)



if __name__ == "__main__":
    client_id = int(sys.argv[1])
    comm_manager = GRPCStreamCommManager("127.0.0.1", "8890",server_ip="127.0.0.1", client_id=client_id, client_num=10)
    comm_manager.handle_receive_message()
    