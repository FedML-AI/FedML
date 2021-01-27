import logging

from typing import List
from concurrent import futures
import threading

import grpc

from ..gRPC import grpc_comm_manager_pb2_grpc, grpc_comm_manager_pb2

lock = threading.Lock()

from FedML.fedml_core.distributed.communication.base_com_manager import BaseCommunicationManager
from FedML.fedml_core.distributed.communication.message import Message
from FedML.fedml_core.distributed.communication.observer import Observer
from FedML.fedml_core.distributed.communication.gRPC.grpc_server import GRPCCOMMServicer


class GRPCCommManager(BaseCommunicationManager):

    def __init__(self, host, port, topic='fedml', client_id=0, client_num=0):
        self.host = host
        self.port = str(port)
        self._topic = topic
        self.client_id = client_id
        self.client_num = client_num
        self._observers: List[Observer] = []

        if client_id == 0:
            self.node_type = "server"
        else:
            self.node_type = "client"

        self.grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=client_num))
        self.grpc_servicer = GRPCCOMMServicer(host, port, client_num, client_id)
        grpc_comm_manager_pb2_grpc.add_gRPCCommManagerServicer_to_server(
            self.grpc_servicer,
            self.grpc_server
        )

        self.grpc_server.add_insecure_port("{}:{}".format(host, port))

        self.grpc_server.start()
        print("server started. Listening on port " + str(port))

    def send_message(self, msg: Message):
        payload = msg.to_json()

        receiver_id = msg.get_receiver_id()

        if receiver_id == 0:
            channel_url = '{}:{}'.format('67.199.133.186', str(50000 + receiver_id))
        elif receiver_id == 1:
            channel_url = '{}:{}'.format('67.199.133.186', str(50000 + receiver_id))
        elif receiver_id == 2:
            channel_url = '{}:{}'.format('117.161.90.229', str(50000 + receiver_id))

        channel = grpc.insecure_channel(channel_url)
        stub = grpc_comm_manager_pb2_grpc.gRPCCommManagerStub(channel)

        request = grpc_comm_manager_pb2.CommRequest()
        logging.info("sending message to port " + str(50000 + int(msg.get_receiver_id())))

        request.client_id = self.client_id

        request.message = payload

        stub.sendMessage(request)
        logging.info("sent")

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        thread = threading.Thread(target=self.message_handling_subroutine)
        thread.start()

    def message_handling_subroutine(self):
        while True:
            if self.grpc_servicer.message_q.qsize() > 0:
                lock.acquire()
                msg_params_string = self.grpc_servicer.message_q.get()
                msg_params = Message()
                msg_params.init_from_json_string(msg_params_string)
                msg_type = msg_params.get_type()
                for observer in self._observers:
                    observer.receive_message(msg_type, msg_params)
                lock.release()

    def stop_receive_message(self):
        pass

    def notify(self, message: Message):
        msg_type = message.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, message)
