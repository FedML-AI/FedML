import csv
import logging
import os
import threading
import time
from typing import List

from torch.distributed import rpc

from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent
from .trpc_server import TRPCCOMMServicer
from .utils import WORKER_NAME, set_device_map
from ..base_com_manager import BaseCommunicationManager
from ..constants import CommunicationConstants
from ..message import Message
from ..observer import Observer

lock = threading.Lock()


class TRPCCommManager(BaseCommunicationManager):
    def __init__(self, trpc_master_config_path, process_id=0, world_size=0, args=None):
        logging.info("using TRPC backend")
        with open(trpc_master_config_path, newline="") as csv_file:
            csv_reader = csv.reader(csv_file)
            # skip header line
            next(csv_reader)
            master_address, master_port = next(csv_reader)
        self.master_address = master_address
        self.master_port = master_port
        self.process_id = process_id
        self.rank = process_id
        self.world_size = world_size
        self._observers: List[Observer] = []
        self.args = args
        if process_id == 0:
            self.node_type = "server"
        else:
            self.node_type = "client"

        logging.info(f"Worker rank {process_id} initializing RPC")

        self.trpc_servicer = TRPCCOMMServicer(master_address, master_port, self.world_size, process_id)
        logging.info(os.getcwd())

        os.environ["MASTER_ADDR"] = self.master_address
        os.environ["MASTER_PORT"] = self.master_port

        self._init_torch_rpc_tp(master_address, master_port, process_id, self.world_size)
        self.is_running = True
        logging.info("server started. master address: " + str(master_address))

    def _init_torch_rpc_tp(
        self, master_addr, master_port, worker_idx, worker_num,
    ):
        # https://github.com/pytorch/pytorch/issues/55615
        # [BC-Breaking][RFC] Retire ProcessGroup Backend for RPC #55615
        str_init_method = "tcp://" + str(master_addr) + ":" + str(master_port)
        logging.info("str_init_method = {}".format(str_init_method))
        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16, rpc_timeout=1800, init_method=str_init_method, _transports=["uv"],
        )
        if self.args.enable_cuda_rpc:
            set_device_map(options, worker_idx, self.args.cuda_rpc_gpu_mapping)

        rpc.init_rpc(
            WORKER_NAME.format(worker_idx),
            backend=rpc.BackendType.TENSORPIPE,
            rank=worker_idx,
            world_size=worker_num,
            rpc_backend_options=options,
        )
        logging.info("_init_torch_rpc_tp finished.")

    def send_message(self, msg: Message):
        receiver_id = msg.get_receiver_id()

        logging.info("sending message to {}".format(receiver_id))

        # Should I wait?
        tick = time.time()
        rpc.rpc_sync(
            WORKER_NAME.format(receiver_id), TRPCCOMMServicer.sendMessage, args=(self.process_id, msg),
        )
        MLOpsProfilerEvent.log_to_wandb({"Comm/send_delay": time.time() - tick})
        logging.debug("sent")

    def add_observer(self, observer: Observer):
        self._observers.append(observer)

    def remove_observer(self, observer: Observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        thread = threading.Thread(target=self.message_handling_subroutine)
        thread.start()
        self._notify_connection_ready()

    def message_handling_subroutine(self):
        start_listening_time = time.time()
        MLOpsProfilerEvent.log_to_wandb({"ListenStart": start_listening_time})
        while self.is_running:
            if self.trpc_servicer.message_q.qsize() > 0:
                lock.acquire()
                message_handler_start_time = time.time()
                msg = self.trpc_servicer.message_q.get()
                self.notify(msg)
                MLOpsProfilerEvent.log_to_wandb({"BusyTime": time.time() - message_handler_start_time})
                lock.release()
        MLOpsProfilerEvent.log_to_wandb({"TotalTime": time.time() - start_listening_time})
        return

    def stop_receive_message(self):
        rpc.shutdown()
        self.is_running = False

    def notify(self, message: Message):
        msg_type = message.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, message)

    def _notify_connection_ready(self):
        msg_params = Message()
        msg_params.sender_id = self.rank
        msg_params.receiver_id = self.rank
        msg_type = CommunicationConstants.MSG_TYPE_CONNECTION_IS_READY
        for observer in self._observers:
            observer.receive_message(msg_type, msg_params)
