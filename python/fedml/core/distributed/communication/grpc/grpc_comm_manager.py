import os
import pickle
import threading
from concurrent import futures
from typing import List

import grpc

from ..grpc import grpc_comm_manager_pb2_grpc, grpc_comm_manager_pb2

lock = threading.Lock()

from ...communication.base_com_manager import BaseCommunicationManager
from ...communication.message import Message
from ...communication.observer import Observer
from ..constants import CommunicationConstants

from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent

import time

# Check Service or serve?
from ...communication.grpc.grpc_server import GRPCCOMMServicer

import logging

import csv


class GRPCCommManager(BaseCommunicationManager):
    def __init__(
        self,
        host,
        port,
        ip_config_path,
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

        if client_id == 0:
            self.node_type = "server"
            logging.info("############# THIS IS FL SERVER ################")
        else:
            self.node_type = "client"
            logging.info("------------- THIS IS FL CLIENT ----------------")
        self.opts = [
            ("grpc.max_send_message_length", 1000 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1000 * 1024 * 1024),
            ("grpc.enable_http_proxy", 0),
        ]
        self.grpc_server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=client_num),
            options=self.opts,
        )
        self.grpc_servicer = GRPCCOMMServicer(host, port, client_num, client_id)
        grpc_comm_manager_pb2_grpc.add_gRPCCommManagerServicer_to_server(
            self.grpc_servicer, self.grpc_server
        )
        logging.info(os.getcwd())
        self.ip_config = self._build_ip_table(ip_config_path)

        # starts a grpc_server on local machine using ip address "0.0.0.0"
        self.grpc_server.add_insecure_port("{}:{}".format("0.0.0.0", port))

        self.grpc_server.start()
        self.is_running = True
        logging.info("grpc server started. Listening on port " + str(port))

    def send_message(self, msg: Message):
        logging.info("msg = {}".format(msg))
        # payload = msg.to_json()

        logging.info("pickle.dumps(msg) START")
        pickle_dump_start_time = time.time()
        msg_pkl = pickle.dumps(msg)
        MLOpsProfilerEvent.log_to_wandb({"PickleDumpsTime": time.time() - pickle_dump_start_time})
        logging.info("pickle.dumps(msg) END")

        receiver_id = msg.get_receiver_id()
        PORT_BASE = CommunicationConstants.GRPC_BASE_PORT
        # lookup ip of receiver from self.ip_config table
        receiver_ip = self.ip_config[str(receiver_id)]
        channel_url = "{}:{}".format(receiver_ip, str(PORT_BASE + receiver_id))

        channel = grpc.insecure_channel(channel_url, options=self.opts)
        stub = grpc_comm_manager_pb2_grpc.gRPCCommManagerStub(channel)

        request = grpc_comm_manager_pb2.CommRequest()
        logging.info("sending message to {}".format(channel_url))

        request.client_id = self.client_id

        request.message = msg_pkl

        tick = time.time()
        stub.sendMessage(request)
        MLOpsProfilerEvent.log_to_wandb({"Comm/send_delay": time.time() - tick})
        logging.debug("sent successfully")
        channel.close()

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        self._notify_connection_ready()
        self.message_handling_subroutine()

        # Cannont run message_handling_subroutine in new thread
        # Related https://stackoverflow.com/a/70705165
        
        # thread = threading.Thread(target=self.message_handling_subroutine)
        # thread.start()

    def message_handling_subroutine(self):
        start_listening_time = time.time()
        MLOpsProfilerEvent.log_to_wandb({"ListenStart": start_listening_time})
        while self.is_running:
            if self.grpc_servicer.message_q.qsize() > 0:
                lock.acquire()
                busy_time_start_time = time.time()
                msg_pkl = self.grpc_servicer.message_q.get()
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
            time.sleep(0.0001)
        MLOpsProfilerEvent.log_to_wandb({"TotalTime": time.time() - start_listening_time})
        return

    def stop_receive_message(self):
        self.grpc_server.stop(None)
        self.is_running = False

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

    def _build_ip_table(self, path):
        ip_config = dict()
        with open(path, newline="") as csv_file:
            csv_reader = csv.reader(csv_file)
            # skip header line
            next(csv_reader)

            for row in csv_reader:
                receiver_id, receiver_ip = row
                ip_config[receiver_id] = receiver_ip
        return ip_config
