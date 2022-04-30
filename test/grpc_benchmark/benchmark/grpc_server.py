import pickle
from concurrent import futures

import grpc

import benchmark_pb2
import benchmark_pb2_grpc
from common import (
    identity,
    identity_script,
    heavy,
    heavy_script,
    identity_cuda,
    identity_script_cuda,
    heavy_cuda,
    heavy_script_cuda,
    stamp_time,
    compute_delay,
)

funcs = {
    "identity": identity,
    "identity_script": identity_script,
    "heavy": heavy,
    "heavy_script": heavy_script,
    "identity_cuda": identity_cuda,
    "identity_script_cuda": identity_script_cuda,
    "heavy_cuda": heavy_cuda,
    "heavy_script_cuda": heavy_script_cuda,
    "stamp_time": stamp_time,
    "compute_delay": compute_delay,
}

MAX_MESSAGE_LENGTH = 10000 * 10000 * 10


class Server(benchmark_pb2_grpc.GRPCBenchmarkServicer):
    def __init__(self, server_address):
        self.server_address = server_address
        self.future = futures.Future()

    def meta_run(self, request, context):
        name, tensor, cuda = pickle.loads(request.data)
        if cuda:
            tensor = tensor.cuda(0)
        return benchmark_pb2.Response(data=pickle.dumps(funcs[name](tensor)))

    def terminate(self, request, context):
        self.future.set_result(0)
        return benchmark_pb2.EmptyMessage()

    def run(self):
        server = grpc.server(
            futures.ThreadPoolExecutor(),
            options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ],
        )

        benchmark_pb2_grpc.add_GRPCBenchmarkServicer_to_server(self, server)

        server.add_insecure_port(self.server_address)
        server.start()
        self.future.result()


def run(addr="localhost", port="29500"):
    server = Server(f"{addr}:{port}")
    server.run()
