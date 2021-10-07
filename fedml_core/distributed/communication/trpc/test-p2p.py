import os
import sys
import torch
import time
import decimal
from torch.distributed import rpc
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../../")))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.getcwd(), "../../../../../FedML")))

from fedml_core.distributed.communication.trpc.trpc_server import TRPCCOMMServicer
from fedml_core.distributed.communication.trpc.trpc_comm_manager import TRPCCommManager
from fedml_core.distributed.communication.observer import Observer
from fedml_core.distributed.communication.message import Message
from fedml_core.distributed.communication.base_com_manager import BaseCommunicationManager

from mpi4py import MPI

np.set_printoptions(formatter={'float_kind': '{:f}'.format})

def run_worker():
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = 2
    if rank == 1:
        com_manager_client = TRPCCommManager("./trpc_master_config.csv", rank, world_size)
        # start = time.time()
        # tensor = torch.ones(1000, 1000)
        # # tensor.cuda(7)
        # message = Message(type="test", sender_id=rank, receiver_id="1")
        # message.add_params("THE_TENSOR", tensor)
        # TRPCCOMMServicer.sendMessage("worker0", message)
        message_values = []
        # message = Message(type="test", sender_id=rank, receiver_id="1")
        # message2 = Message(type="test", sender_id=rank, receiver_id="1")
        # message.add_params("THE_TENSOR", tensor)
        size = 100
        for i in range(100):
            # print("Iteration")
            tensor = torch.ones(size, size)
            tensor.to(7)
            start = time.time()
            TRPCCOMMServicer.sendMessageTest1("worker0", tensor)
            end = time.time()
            duration = end - start
            message_values.append(duration)
                # print(f"Message tensor size={size} duration={str(duration)}", flush=True)
        print(np.sum(message_values))
        # print("mean message: " + str(np.mean(message_values)))
        # print("mean single tensor: " + str(decimal.Decimal(sum(sinle_tensor_values) / len(sinle_tensor_values))))
        # ret = rpc.rpc_sync("worker1", TRPCCOMMServicer., args=(torch.ones(2), torch.ones(2)))
    else:
        # parameter server does nothing
        com_manager_client = TRPCCommManager("./trpc_master_config.csv", rank, world_size)

    rpc.shutdown()





# def run_worker_cuda_aware(rank, world_size):
#     r"""
#     A wrapper function that initializes RPC, calls the function, and shuts down
#     RPC.
#     """
#     if rank == 1:
#         com_manager_client = TRPCCommManager("./trpc_master_config.csv", rank, world_size)
#         start = time.time()
#         tensor = torch.ones(1000, 1000)
#         message = Message(type="test", sender_id=rank, receiver_id="1")
#         message.add_params("THE_TENSOR", tensor)
#         TRPCCOMMServicer.sendMessage("worker0", message)
#         message_values = []
#         message = Message(type="test", sender_id=rank, receiver_id="1")
#         message2 = Message(type="test", sender_id=rank, receiver_id="1")
#         message.add_params("THE_TENSOR", tensor)
#         for i in range(100):
#             print("###############################")
#             print("Measuring for Single Message")
#             for size in [100, 1000, 10000]:

#                 # for size in [100, 1000]:
#                 print(f"======= size = {size} =====")
#                 tensor = torch.ones(size, size)
#                 start = time.time()
#                 TRPCCOMMServicer.sendMessageTest1("worker0", message)
#                 end = time.time()
#                 duration = end - start
#                 message_values.append(duration)
#                 # print(f"Message tensor size={size} duration={str(duration)}", flush=True)

#             print("###############################")
#             print("Measuring for Message with separate Tensor")
#             sinle_tensor_values = []
#             start = time.time()
#             for size in [100, 1000, 10000]:

#                 # for size in [100, 1000]:
#                 print(f"======= size = {size} =====")
#                 tensor = torch.ones(size, size)
#                 # message = Message(type="test", sender_id=rank, receiver_id="1")
#                 # message.add_params("THE_TENSOR", tensor)
#                 start = time.time()
#                 TRPCCOMMServicer.sendMessageTest2("worker0", message2.get_params(), tensor)
#                 end = time.time()
#                 duration = end - start
#                 # print(f"Single tensor size={size} duration={str(duration)}", flush=True)
#                 sinle_tensor_values.append(duration)

#         print("mean message: " + str(decimal.Decimal(sum(message_values) / len(message_values))))
#         print("mean single tensor: " + str(decimal.Decimal(sum(sinle_tensor_values) / len(sinle_tensor_values))))
#         # ret = rpc.rpc_sync("worker1", TRPCCOMMServicer., args=(torch.ones(2), torch.ones(2)))
#     else:
#         # parameter server does nothing
#         com_manager_client = TRPCCommManager("./trpc_master_config.csv", rank, world_size)

#     rpc.shutdown()



if __name__ == "__main__":
    run_worker()
    # world_size = 2
    # # run_worker(0,1)
    # mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
