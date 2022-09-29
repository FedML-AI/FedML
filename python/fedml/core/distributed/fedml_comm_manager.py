import json
import logging
import grpc
from abc import abstractmethod

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
        try:
            handler_callback_func = self.message_handler_dict[msg_type]
            handler_callback_func(msg_params)
        except KeyError:
            raise Exception(
                "KeyError. msg_type = {}. Please check whether you launch the server or client with the correct args.rank".format(
                    msg_type
                )
            )

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

    def get_training_mqtt_s3_config(self):
        mqtt_config = None
        s3_config = None
        if hasattr(self.args, "customized_training_mqtt_config") and self.args.customized_training_mqtt_config != "":
            mqtt_config = self.args.customized_training_mqtt_config
        if hasattr(self.args, "customized_training_s3_config") and self.args.customized_training_s3_config != "":
            s3_config = self.args.customized_training_s3_config
        if mqtt_config is None or s3_config is None:
            mqtt_config_from_cloud, s3_config_from_cloud = MLOpsConfigs.get_instance(self.args).fetch_configs()
            if mqtt_config is None:
                mqtt_config = mqtt_config_from_cloud
            if s3_config is None:
                s3_config = s3_config_from_cloud

        return mqtt_config, s3_config

    def get_training_mqtt_ipfs_config(self):
        mqtt_config = None
        ipfs_config = None
        if hasattr(self.args, "customized_training_mqtt_config") and self.args.customized_training_mqtt_config != "":
            mqtt_config = self.args.customized_training_mqtt_config
        if hasattr(self.args, "customized_training_ipfs_config") and self.args.customized_training_ipfs_config != "":
            ipfs_config = self.args.customized_training_ipfs_config
        if mqtt_config is None or ipfs_config is None:
            mqtt_config_from_cloud, ipfs_config_from_cloud = MLOpsConfigs.get_instance(self.args).fetch_ipfs_configs()
            if mqtt_config is None:
                mqtt_config = mqtt_config_from_cloud
            if ipfs_config is None:
                ipfs_config = s3_config_from_cloud

        return mqtt_config, ipfs_config

    def _init_manager(self):

        if self.backend == "MPI":
            from .communication.mpi.com_manager import MpiCommunicationManager

            self.com_manager = MpiCommunicationManager(self.comm, self.rank, self.size)
        elif self.backend == "MQTT_S3":
            from .communication.mqtt_s3.mqtt_s3_multi_clients_comm_manager import MqttS3MultiClientsCommManager

            mqtt_config, s3_config = self.get_training_mqtt_s3_config()

            self.com_manager = MqttS3MultiClientsCommManager(
                mqtt_config,
                s3_config,
                topic=str(self.args.run_id),
                client_rank=self.rank,
                client_num=self.size,
                args=self.args,
            )
        elif self.backend == "MQTT_S3_MNN":
            from .communication.mqtt_s3_mnn.mqtt_s3_comm_manager import MqttS3MNNCommManager

            mqtt_config, s3_config = self.get_training_mqtt_s3_config()

            self.com_manager = MqttS3MNNCommManager(
                mqtt_config,
                s3_config,
                topic=str(self.args.run_id),
                client_id=self.rank,
                client_num=self.size,
                args=self.args,
            )
        elif self.backend == "MQTT_IPFS":
            from .communication.mqtt_ipfs.mqtt_ipfs_comm_manager import MqttIpfsCommManager

            mqtt_config, ipfs_config = self.get_training_mqtt_ipfs_config()

            self.com_manager = MqttIpfsCommManager(
                mqtt_config,
                ipfs_config,
                topic=str(self.args.run_id),
                client_rank=self.rank,
                client_num=self.size,
                args=self.args,
            )
        elif self.backend == "GRPC":
            from .communication.grpc.grpc_comm_manager import GRPCCommManager
            from .communication.grpc.grpc_secure_comm_manager import GRPCSecureCommManager

            HOST = "0.0.0.0"
            PORT = CommunicationConstants.GRPC_BASE_PORT + self.rank
            if self.args.grpc_certificate != "" and self.args.grpc_private_key != "":
                private_key = open(self.args.grpc_private_key, 'rb').read()
                certificate = open(self.args.grpc_certificate, 'rb').read()
                credentials = grpc.ssl_server_credentials([(
                    private_key,
                    certificate
                )])
                ca_certificate = open(self.args.grpc_trusted_ca, 'rb').read()
                ca_credentials = grpc.ssl_channel_credentials(ca_certificate)
                self.com_manager = GRPCSecureCommManager(
                    HOST, PORT, credentials, ca_credentials,
                    ip_config_path=self.args.grpc_ipconfig_path, client_id=self.rank, client_num=self.size
                )
            else:
                self.com_manager = GRPCCommManager(
                    HOST, PORT, ip_config_path=self.args.grpc_ipconfig_path, client_id=self.rank, client_num=self.size,
                )
        elif self.backend == "TRPC":
            from .communication.trpc.trpc_comm_manager import TRPCCommManager

            self.com_manager = TRPCCommManager(
                self.args.trpc_master_config_path, process_id=self.rank, world_size=self.size + 1, args=self.args,
            )
        else:
            if self.com_manager is None:
                raise Exception("no such backend: {}. Please check the comm_backend spelling.".format(self.backend))
            else:
                logging.info("using self-defined communication backend")

        self.com_manager.add_observer(self)
