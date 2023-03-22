import ctypes
import logging
import threading
import traceback

from ..message import Message


class MPIReceiveThread(threading.Thread):
    def __init__(self, comm, rank, size, name, q):
        super(MPIReceiveThread, self).__init__()
        self._stop_event = threading.Event()
        self.comm = comm
        self.rank = rank
        self.size = size
        self.name = name
        self.q = q

    def run(self):
        logging.debug(
            "Starting Thread:" + self.name + ". Process ID = " + str(self.rank)
        )
        while True:
            try:
                msg = self.comm.recv()
                # Ugly delete comments
                # msg_str = self.comm.recv()
                # msg = Message()
                # msg.init(msg_str)
                self.q.put(msg)
            except Exception:
                traceback.print_exc()
                raise Exception("MPI failed!")

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            thread_id, ctypes.py_object(SystemExit)
        )
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print("Exception raise failure")
