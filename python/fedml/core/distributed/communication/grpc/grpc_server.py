from ..grpc import grpc_comm_manager_pb2, grpc_comm_manager_pb2_grpc
import queue
import threading
import logging
from ...communication.utils import log_communication_tock
from time import time

lock = threading.Lock()


class GRPCCOMMServicer(grpc_comm_manager_pb2_grpc.gRPCCommManagerServicer):
    def __init__(self, host, port, client_num, client_id):
        # host is the ip address of server
        self.host = host
        self.port = port
        self.client_num = client_num
        self.client_id = client_id

        if self.client_id == 0:
            self.node_type = "server"
        else:
            self.node_type = "client"

        self.message_q = queue.Queue()

    def sendMessage(self, request, context):
        context_ip = context.peer().split(":")[1]
        logging.info(
            "client_{} got something from client_{} from ip address {}".format(
                self.client_id, request.client_id, context_ip
            )
        )

        response = grpc_comm_manager_pb2.CommResponse()
        # response.message = "message received"
        lock.acquire()
        self.message_q.put(request.message)
        lock.release()
        return response

    def handleReceiveMessage(self, request, context):
        pass
