import queue
import threading
import logging

lock = threading.Lock()


class TRPCCOMMServicer:
    _instance = None

    def __new__(cls, master_address, master_port, client_num, client_id):
        """
        Create a new instance of the TRPCCOMMServicer class if it does not exist, otherwise return the existing instance.

        Args:
            master_address (str): The address of the RPC master.
            master_port (str): The port of the RPC master.
            client_num (int): The total number of clients.
            client_id (int): The ID of the current client.

        Returns:
            TRPCCOMMServicer: An instance of the TRPCCOMMServicer class.
        """
        cls.master_address = None
        cls.master_port = None
        cls.client_num = None
        cls.client_id = None
        cls.node_type = None
        cls.message_q = None

        if cls._instance is None:
            cls._instance = super(TRPCCOMMServicer, cls).__new__(cls)
            cls._instance.master_address = master_address
            cls._instance.master_port = master_port
            cls._instance.client_num = client_num
            cls._instance.client_id = client_id
            if cls._instance.client_id == 0:
                cls._instance.node_type = "server"
            else:
                cls._instance.node_type = "client"
            cls._instance.message_q = queue.Queue()
            # Put any initialization here.
        return cls._instance

    def receiveMessage(self, client_id, message):
        """
        Receive a message from another client.

        Args:
            client_id (int): The ID of the client sending the message.
            message (Message): The received message.

        Returns:
            str: A response indicating that the message was received.
        """
        logging.info(
            "client_{} got something from client_{}".format(
                self.client_id,
                client_id,
            )
        )
        print(
            "client_{} got something from client_{}".format(
                self.client_id,
                client_id,
            )
        )
        response = "message received"
        lock.acquire()
        self.message_q.put(message)
        lock.release()
        return response

    @classmethod
    def sendMessage(cls, clint_id, message):
        """
        Send a message to another client.

        Args:
            clint_id (int): The ID of the target client.
            message (Message): The message to be sent.

        Returns:
            None
        """
        cls._instance.receiveMessage(clint_id, message)