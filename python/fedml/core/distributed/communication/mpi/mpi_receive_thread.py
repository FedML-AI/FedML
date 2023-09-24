import ctypes
import logging
import threading
import traceback

from ..message import Message


class MPIReceiveThread(threading.Thread):
    """
    MPI Receive Thread.

    This thread is responsible for receiving messages using MPI.
    """

    def __init__(self, comm, rank, size, name, q):
        """
        Initialize the MPI Receive Thread.

        Args:
            comm: The MPI communicator.
            rank: The rank of the current process.
            size: The total number of processes in the communicator.
            name: The name of the thread.
            q: The message queue to store received messages.
        """
        super(MPIReceiveThread, self).__init__()
        self._stop_event = threading.Event()
        self.comm = comm
        self.rank = rank
        self.size = size
        self.name = name
        self.q = q

    def run(self):
        """
        Run the MPI Receive Thread.

        This method continuously listens for incoming messages and puts them into the message queue.
        """
        logging.debug(
            "Starting Thread:" + self.name + ". Process ID = " + str(self.rank)
        )
        while True:
            try:
                msg = self.comm.recv()
                self.q.put(msg)
            except Exception:
                traceback.print_exc()
                raise Exception("MPI failed!")

    def stop(self):
        """
        Stop the MPI Receive Thread.
        """
        self._stop_event.set()

    def stopped(self):
        """
        Check if the MPI Receive Thread is stopped.

        Returns:
            bool: True if the thread is stopped, False otherwise.
        """
        return self._stop_event.is_set()

    def get_id(self):
        """
        Get the ID of the thread.

        Returns:
            int: The ID of the thread.
        """
        # returns id of the respective thread
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        """
        Raise an exception in the MPI Receive Thread to stop it.
        """
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            thread_id, ctypes.py_object(SystemExit)
        )
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print("Exception raise failure")
