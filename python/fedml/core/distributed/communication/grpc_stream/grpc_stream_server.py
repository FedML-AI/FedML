from . import grpc_stream_comm_manager_pb2, grpc_stream_comm_manager_pb2_grpc
import queue
import threading
import logging
# from ...communication.utils import log_communication_tock
from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent
import time

lock = threading.Lock()
    

class GRPCStreamCOMMServicer(grpc_stream_comm_manager_pb2_grpc.GRPCStreamCommManager):
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
        response = grpc_stream_comm_manager_pb2.Empty()
        self.queue_manager.add_to_received_messages_queue(request.message)
        return response


    def Connect(self, request, context):
        client_id = request.client_id 
        self.queue_manager.create_send_messages_queue(client_id)
        logging.info(f"Client {client_id} connected")
        for msg in self.queue_manager.get_client_send_messages_iterator(client_id):
            tick = time.time()
            yield msg
            MLOpsProfilerEvent.log_to_wandb({"Comm/send_delay": time.time() - tick})
        return

