from FedML.fedml_core.distributed.communication.gRPC import grpc_comm_manager_pb2, grpc_comm_manager_pb2_grpc
import queue
import threading
lock = threading.Lock()

class GRPCCOMMServicer(grpc_comm_manager_pb2_grpc.gRPCCommManagerServicer):
    def __init__(self, host, port, client_num, client_id):
        self.host = host
        self.port = port
        self.client_num = client_num
        self.client_id = client_id

        self.message_q = queue.Queue()

    def sendMessage(self, request, context):
        print("client_{} got something from client_{}".format(
            self.client_id,
            request.client_id
        ))
        response = grpc_comm_manager_pb2.CommResponse()
        response.message = "message received"
        lock.acquire()
        self.message_q.put(request.message)
        lock.release()
        return response

    def handleReceiveMessage(self, request, context):
        pass