import ctypes
import logging
import threading
import time
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
        # Infinite loop.
        while True:            
            try:
                # Loop till a new message arrives.
                while not self.comm.Iprobe():
                    time.sleep(0.001)
                    if self._stop_event.is_set():
                        return           
                msg = self.comm.recv() # Blocking-Call!            
                self.q.put(msg)
            except Exception:
                traceback.print_exc()
                raise Exception("MPI failed!")

    def stop(self):
        self._stop_event.set()

    def get_id(self):
        # returns id of the respective thread.
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id
