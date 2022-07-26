import logging
from abc import abstractmethod

from google.api.logging_pb2 import Logging

from .communication.base_com_manager import BaseCommunicationManager
from .communication.constants import CommunicationConstants
from .communication.observer import Observer
from ..mlops.mlops_configs import MLOpsConfigs


class FedMLCommManager(Observer):
    def __init__(self, args, comm=None, rank=0, size=0, backend="MPI"):
        self.args = args
        self.size = size
        self.rank = int(rank)
        self.backend = backend
        self.comm = comm
        self.com_manager = None
        self.message_handler_dict = dict()
        self._init_manager()

    def register_comm_manager(self, comm_manager: BaseCommunicationManager):
        self.com_manager = comm_manager

    def run(self):
        self.register_message_receive_handlers()
        logging.info("running")
        self.com_manager.handle_receive_message()
        logging.info("finished...")

    def get_sender_id(self):
        return self.rank

    def receive_message(self, msg_type, msg_params) -> None:
        logging.info(
            "receive_message. msg_type = %s, sender_id = %d, receiver_id = %d"
            % (str(msg_type), msg_params.get_sender_id(), msg_params.get_receiver_id())
        )
        handler_callback_func = self.message_handler_dict[msg_type]
        handler_callback_func(msg_params)

    def send_message(self, message):
        self.com_manager.send_message(message)

    def send_message_json(self, topic_name, json_message):
        self.com_manager.send_message_json(topic_name, json_message)

    @abstractmethod
    def register_message_receive_handlers(self) -> None:
        pass

    def register_message_receive_handler(self, msg_type, handler_callback_func):
        self.message_handler_dict[msg_type] = handler_callback_func

    def finish(self):
        logging.info("__finish")
        if self.backend == "MPI":
            from mpi4py import MPI

            MPI.COMM_WORLD.Abort()
        elif self.backend == "MQTT":
            self.com_manager.stop_receive_message()
        elif self.backend == "MQTT_S3":
            self.com_manager.stop_receive_message()
        elif self.backend == "MQTT_S3_MNN":
            self.com_manager.stop_receive_message()
        elif self.backend == "GRPC":
            self.com_manager.stop_receive_message()
        elif self.backend == "TRPC":
            self.com_manager.stop_receive_message()

    def _init_manager(self):

        if self.backend == "MPI":
            from .communication.mpi.com_manager import MpiCommunicationManager

            self.com_manager = MpiCommunicationManager(self.comm, self.rank, self.size)
        elif self.backend == "MQTT":
            from .communication.mqtt.mqtt_comm_manager import MqttCommManager

            HOST = "0.0.0.0"
            # HOST = "broker.emqx.io"
            PORT = 1883
            self.com_manager = MqttCommManager(HOST, PORT, client_id=self.rank, client_num=self.size - 1)
        elif self.backend == "MQTT_S3":
            from .communication.mqtt_s3.mqtt_s3_multi_clients_comm_manager import MqttS3MultiClientsCommManager

            mqtt_config, s3_config = MLOpsConfigs.get_instance(self.args).fetch_configs()
            self.args.mqtt_config_path = mqtt_config
            self.args.s3_config_path = s3_config
            self.com_manager = MqttS3MultiClientsCommManager(
                self.args.mqtt_config_path,
                self.args.s3_config_path,
                topic=str(self.args.run_id),
                client_rank=self.rank,
                client_num=self.size,
                args=self.args,
            )

        elif self.backend == "MQTT_S3_MNN":
            from .communication.mqtt_s3_mnn.mqtt_s3_comm_manager import MqttS3MNNCommManager

            mqtt_config, s3_config = MLOpsConfigs.get_instance(self.args).fetch_configs()
            self.args.mqtt_config_path = mqtt_config
            self.args.s3_config_path = s3_config
            self.com_manager = MqttS3MNNCommManager(
                self.args.mqtt_config_path,
                self.args.s3_config_path,
                topic=str(self.args.run_id),
                client_id=self.rank,
                client_num=self.size,
                args=self.args,
            )

        elif self.backend == "GRPC":
            from .communication.grpc.grpc_comm_manager import GRPCCommManager

            HOST = "0.0.0.0"
            PORT = CommunicationConstants.GRPC_BASE_PORT + self.rank
            self.com_manager = GRPCCommManager(
                HOST, PORT, ip_config_path=self.args.grpc_ipconfig_path, client_id=self.rank, client_num=self.size,
            )
        elif self.backend == "TRPC":
            from .communication.trpc.trpc_comm_manager import TRPCCommManager

            self.com_manager = TRPCCommManager(
                self.args.trpc_master_config_path, process_id=self.rank, world_size=self.size + 1, args=self.args,
            )
        else:
            Logging.info("using self-defined communication backend")

        self.com_manager.add_observer(self)
