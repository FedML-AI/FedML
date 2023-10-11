import json
import logging
from abc import abstractmethod

from .communication.base_com_manager import BaseCommunicationManager
from .communication.constants import CommunicationConstants
from .communication.observer import Observer
from ..mlops.mlops_configs import MLOpsConfigs


class FedMLCommManager(Observer):
    """
    Communication manager for Federated Machine Learning (FedML).

    Args:
        args: Command-line arguments.
        comm: The communication backend.
        rank: The rank of the current node.
        size: The total number of nodes in the communication group.
        backend: The communication backend used (e.g., "MPI", "MQTT", "MQTT_S3").

    Attributes:
        args: Command-line arguments.
        size: The total number of nodes in the communication group.
        rank: The rank of the current node.
        backend: The communication backend used.
        comm: The communication object.
        com_manager: The communication manager.
        message_handler_dict: A dictionary to register message handlers.

    Methods:
        register_comm_manager(comm_manager): Register a communication manager.
        run(): Start the communication manager.
        get_sender_id(): Get the sender's ID.
        receive_message(msg_type, msg_params): Receive a message and handle it.
        send_message(message): Send a message.
        send_message_json(topic_name, json_message): Send a JSON message.
        register_message_receive_handlers(): Register message receive handlers.
        register_message_receive_handler(msg_type, handler_callback_func): Register a message receive handler.
        finish(): Finish the communication manager.
        get_training_mqtt_s3_config(): Get MQTT and S3 configurations for training.
        get_training_mqtt_web3_config(): Get MQTT and Web3 configurations for training.
        get_training_mqtt_thetastore_config(): Get MQTT and Thetastore configurations for training.
        _init_manager(): Initialize the communication manager based on the selected backend.
    """

    def __init__(self, args, comm=None, rank=0, size=0, backend="MPI"):
        """
        Initialize the FedMLCommManager.

        Args:
            args: Command-line arguments.
            comm: The communication backend.
            rank: The rank of the current node.
            size: The total number of nodes in the communication group.
            backend: The communication backend used (e.g., "MPI", "MQTT", "MQTT_S3").

        Returns:
            None
        """
        self.args = args
        self.size = size
        self.rank = int(rank)
        self.backend = backend
        self.comm = comm
        self.com_manager = None
        self.message_handler_dict = dict()
        self._init_manager()

    def register_comm_manager(self, comm_manager: BaseCommunicationManager):
        """
        Register a communication manager.

        Args:
            comm_manager (BaseCommunicationManager): The communication manager to register.

        Returns:
            None
        """
        self.com_manager = comm_manager

    def run(self):
        """
        Start the communication manager.

        Returns:
            None
        """
        self.register_message_receive_handlers()
        logging.info("running")
        self.com_manager.handle_receive_message()
        logging.info("finished...")

    def get_sender_id(self):
        """
        Get the sender's ID.

        Returns:
            int: The sender's ID (rank).

        """
        return self.rank

    def receive_message(self, msg_type, msg_params) -> None:
        """
        Receive a message and handle it.

        Args:
            msg_type (str): The type of the received message.
            msg_params: Parameters associated with the received message.

        Returns:
            None
        """

        if msg_params.get_sender_id() == msg_params.get_receiver_id():
            logging.info(
                "communication backend is alive (loop_forever, sender 0 to receiver 0)")
        else:
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
        """
        Send a message.

        Args:
            message: The message to send.

        Returns:
            None
        """
        self.com_manager.send_message(message)

    def send_message_json(self, topic_name, json_message):
        """
        Send a JSON message.

        Args:
            topic_name (str): The name of the message topic.
            json_message: The JSON message to send.

        Returns:
            None
        """
        self.com_manager.send_message_json(topic_name, json_message)

    @abstractmethod
    def register_message_receive_handlers(self) -> None:
        """
        Register message receive handlers.

        This method should be implemented in derived classes.

        Returns:
            None
        """
        pass

    def register_message_receive_handler(self, msg_type, handler_callback_func):
        """
        Register a message receive handler.

        Args:
            msg_type (str): The type of the message to handle.
            handler_callback_func: The callback function to handle the message.

        Returns:
            None
        """
        self.message_handler_dict[msg_type] = handler_callback_func

    def finish(self):
        """
        Finish the communication manager.

        Depending on the backend used, this method may perform specific actions to terminate the communication.

        Returns:
            None
        """
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
        """
        Get MQTT and S3 configurations for training.

        Returns:
            tuple: A tuple containing MQTT configuration and S3 configuration.

        """
        mqtt_config = None
        s3_config = None
        if hasattr(self.args, "customized_training_mqtt_config") and self.args.customized_training_mqtt_config != "":
            mqtt_config = self.args.customized_training_mqtt_config
        if hasattr(self.args, "customized_training_s3_config") and self.args.customized_training_s3_config != "":
            s3_config = self.args.customized_training_s3_config
        if mqtt_config is None or s3_config is None:
            mqtt_config_from_cloud, s3_config_from_cloud = MLOpsConfigs.get_instance(
                self.args).fetch_configs()
            if mqtt_config is None:
                mqtt_config = mqtt_config_from_cloud
            if s3_config is None:
                s3_config = s3_config_from_cloud

        return mqtt_config, s3_config

    def get_training_mqtt_web3_config(self):
        mqtt_config = None
        web3_config = None
        if hasattr(self.args, "customized_training_mqtt_config") and self.args.customized_training_mqtt_config != "":
            mqtt_config = self.args.customized_training_mqtt_config
        if hasattr(self.args, "customized_training_web3_config") and self.args.customized_training_web3_config != "":
            web3_config = self.args.customized_training_web3_config
        if mqtt_config is None or web3_config is None:
            mqtt_config_from_cloud, web3_config_from_cloud = MLOpsConfigs.get_instance(
                self.args).fetch_web3_configs()
            if mqtt_config is None:
                mqtt_config = mqtt_config_from_cloud
            if web3_config is None:
                web3_config = web3_config_from_cloud

        return mqtt_config, web3_config

    def get_training_mqtt_thetastore_config(self):
        mqtt_config = None
        thetastore_config = None
        if hasattr(self.args, "customized_training_mqtt_config") and self.args.customized_training_mqtt_config != "":
            mqtt_config = self.args.customized_training_mqtt_config
        if hasattr(self.args, "customized_training_thetastore_config") and self.args.customized_training_thetastore_config != "":
            thetastore_config = self.args.customized_training_thetastore_config
        if mqtt_config is None or thetastore_config is None:
            mqtt_config_from_cloud, thetastore_config_from_cloud = MLOpsConfigs.get_instance(
                self.args).fetch_thetastore_configs()
            if mqtt_config is None:
                mqtt_config = mqtt_config_from_cloud
            if thetastore_config is None:
                thetastore_config = thetastore_config_from_cloud

        return mqtt_config, thetastore_config

    def _init_manager(self):

        if self.backend == "MPI":
            from .communication.mpi.com_manager import MpiCommunicationManager

            self.com_manager = MpiCommunicationManager(
                self.comm, self.rank, self.size)
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
        elif self.backend == "MQTT_WEB3":
            from .communication.mqtt_web3.mqtt_web3_comm_manager import MqttWeb3CommManager

            mqtt_config, web3_config = self.get_training_mqtt_web3_config()

            self.com_manager = MqttWeb3CommManager(
                mqtt_config,
                web3_config,
                topic=str(self.args.run_id),
                client_rank=self.rank,
                client_num=self.size,
                args=self.args,
            )
        elif self.backend == "MQTT_THETASTORE":
            from .communication.mqtt_thetastore import MqttThetastoreCommManager

            mqtt_config, thetastore_config = self.get_training_mqtt_thetastore_config()

            self.com_manager = MqttThetastoreCommManager(
                mqtt_config,
                thetastore_config,
                topic=str(self.args.run_id),
                client_rank=self.rank,
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
            if self.com_manager is None:
                raise Exception(
                    "no such backend: {}. Please check the comm_backend spelling.".format(self.backend))
            else:
                logging.info("using self-defined communication backend")

        self.com_manager.add_observer(self)
