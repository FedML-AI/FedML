from ..grpc import grpc_comm_manager_stream_pb2, grpc_comm_manager_stream_pb2_grpc
import queue
import threading
import logging
# from ...communication.utils import log_communication_tock
from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent
import time

lock = threading.Lock()
    

class GRPCCOMMServicer(grpc_comm_manager_stream_pb2_grpc.GRPCCommManagerStream):
    def __init__(self, host, port, client_num, client_id, queue_manager):
        # host is the ip address of server
        self.host = host
        self.port = port
        self.client_num = client_num
        self.client_id = client_id
        self.queue_manager = queue_manager

        if self.client_id == 0:
            self.node_type = "server"
        else:
            self.node_type = "client"

        self.recive_queue = queue.Queue()
        self.send_queues = {}

    def SendToServer(self, request, context):
        # context_ip = context.peer().split(":")[1]
        response = grpc_comm_manager_stream_pb2.Empty()
        print("Server Received from: ", request.client_id)
        self.queue_manager.add_to_received_messages_queue(request.message)
        return response


    def Connect(self, request, context):
        # context_ip = context.peer().split(":")[1]
        # peer = context.peer()
        # response = grpc_comm_manager_stream_pb2.CommResponse()
        client_id = request.client_id 
        self.queue_manager.create_send_messages_queue(client_id)
        
        # test_message = grpc_comm_manager_stream_pb2.CommRequest()
        # test_message.client_id = self.client_id
        # test_message.message = bytes("Some Message", "UTF-8")
        # self.queue_manager.add_to_send_messages_queue(client_id, test_message)
        print(f"Client {client_id} connected")

        for msg in self.queue_manager.get_client_send_messages_iterator(client_id):
            tick = time.time()
            yield msg

            # time.sleep(5)
            # self.queue_manager.add_to_send_messages_queue(client_id, test_message)
            # self.queue_manager.stop_iterators()
            # return

            MLOpsProfilerEvent.log_to_wandb({"Comm/send_delay": time.time() - tick})
        return
        # server_message_iterator: Iterator[ServerMessage] = stub.Join(iter(queue.get, None))
        # self.recvied_message_q.put(request.message)
        # return response

