import queue
import threading
import logging

lock = threading.Lock()


class TRPCCOMMServicer:
    # com_manager: TRPCCommManager = None
    _instance = None

    def __new__(cls, master_address, master_port, client_num, client_id):
        if cls._instance is None:
            print("Creating the object")
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

    def receiveMessage(self, clint_id, message):
        print("Recieved")
        logging.info(
            "client_{} got something from client_{}".format(
                self.client_id,
                clint_id,
            )
        )
        print(
            "client_{} got something from client_{}".format(
                self.client_id,
                clint_id,
            )
        )
        response = "message received"
        lock.acquire()
        self.message_q.put(message)
        lock.release()
        return response

    @classmethod
    def sendMessage(cls, clint_id, message):
        cls._instance.receiveMessage(clint_id, message)

    @classmethod
    def sendMessageTest1(cls, clint_id, message):
        return message
        pass
        # cls._instance.receiveMessage(clint_id, message)
        # x = message.get("THE_TENSOR")
        # print("received");

    @classmethod
    def sendMessageTest2(cls, clint_id, arg1, arg2):
        return arg2
        pass
