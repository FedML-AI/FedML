import logging
from abc import abstractmethod

from mpi4py import MPI

from ..communication.grpc.grpc_comm_manager import GRPCCommManager
from ..communication.message import Message
from ..communication.mpi.com_manager import MpiCommunicationManager
from ..communication.mqtt.mqtt_comm_manager import MqttCommManager
from ..communication.mqtt_s3.mqtt_s3_multi_clients_comm_manager import (
    MqttS3MultiClientsCommManager,
)
from ..communication.mqtt_s3.mqtt_s3_status_manager import MqttS3StatusManager
from ..communication.mqtt_s3_mnn.mqtt_s3_comm_manager import MqttS3MNNCommManager
from ..communication.observer import Observer
from ..communication.trpc.trpc_comm_manager import TRPCCommManager


class ClientManager(Observer):
    def __init__(self, args, comm=None, rank=0, size=0, backend="MPI"):
        self.args = args
        self.size = size
        self.rank = int(rank)
        self.backend = backend

        if backend == "MPI":
            self.com_manager = MpiCommunicationManager(
                comm, rank, size, node_type="client"
            )
        elif backend == "MQTT":
            HOST = "0.0.0.0"
            # HOST = "broker.emqx.io"
            PORT = 1883
            self.com_manager = MqttCommManager(
                HOST, PORT, client_id=rank, client_num=size - 1
            )
        elif backend == "MQTT_S3":
            self.com_manager = MqttS3MultiClientsCommManager(
                args.mqtt_config_path,
                args.s3_config_path,
                topic=args.run_id,
                client_rank=rank,
                client_num=size,
                args=args,
            )

            self.com_manager_status = MqttS3StatusManager(
                args.mqtt_config_path, args.s3_config_path, topic=str(args.run_id)
            )
        elif backend == "MQTT_S3_MNN":
            self.com_manager = MqttS3MNNCommManager(
                args.mqtt_config_path,
                args.s3_config_path,
                topic=str(args.run_id),
                client_id=rank,
                client_num=size,
                args=args,
            )
            self.com_manager_status = MqttS3StatusManager(
                args.mqtt_config_path, args.s3_config_path, topic=args.run_id
            )
        elif backend == "GRPC":
            HOST = "0.0.0.0"
            PORT = 8888 + rank
            self.com_manager = GRPCCommManager(
                HOST,
                PORT,
                ip_config_path=args.grpc_ipconfig_path,
                client_id=rank,
                client_num=size,
            )
            if args.using_mlops:
                self.com_manager_status = MqttS3StatusManager(
                    args.mqtt_config_path, args.s3_config_path, topic=args.run_id
                )
        elif backend == "TRPC":
            self.com_manager = TRPCCommManager(
                args.trpc_master_config_path, process_id=rank, world_size=size + 1
            )
            if args.using_mlops:
                self.com_manager_status = MqttS3StatusManager(
                    args.mqtt_config_path, args.s3_config_path, topic=args.run_id
                )
        else:
            self.com_manager = MpiCommunicationManager(
                comm, rank, size, node_type="client"
            )

        self.com_manager.add_observer(self)
        self.message_handler_dict = dict()

    def run(self):
        self.register_message_receive_handlers()
        self.com_manager.handle_receive_message()

    def get_sender_id(self):
        return self.rank

    def receive_message(self, msg_type, msg_params) -> None:
        # logging.info("receive_message. rank_id = %d, msg_type = %s. msg_params = %s" % (
        #     self.rank, str(msg_type), str(msg_params.get_content())))
        handler_callback_func = self.message_handler_dict[msg_type]
        handler_callback_func(msg_params)

    def send_message(self, message):
        msg = Message(
            message.get_type(), message.get_sender_id(), message.get_receiver_id()
        )
        msg.add(Message.MSG_ARG_KEY_TYPE, message.get_type())
        msg.add(Message.MSG_ARG_KEY_SENDER, message.get_sender_id())
        msg.add(Message.MSG_ARG_KEY_RECEIVER, message.get_receiver_id())
        for key, value in message.get_params().items():
            # logging.info("%s == %s" % (key, value))
            msg.add(key, value)
        logging.info("Sending message (type %d) to server" % message.get_type())
        self.com_manager.send_message(msg)
        for key, value in msg.get_params().items():
            # logging.info("%s == %s" % (key, value))
            message.add(key, value)

    @abstractmethod
    def register_message_receive_handlers(self) -> None:
        pass

    def register_message_receive_handler(self, msg_type, handler_callback_func):
        self.message_handler_dict[msg_type] = handler_callback_func

    def finish(self):
        logging.info("__finish client")
        if self.backend == "MPI":
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
