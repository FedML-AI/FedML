import ctypes
import logging
import threading
import time
import traceback

from ..message import Message


class MPISendThread(threading.Thread):
    """
    MPI Send Thread.

    This thread is responsible for sending messages using MPI.
    """

    def __init__(self, comm, rank, size, name, q):
        """
        Initialize the MPI Send Thread.

        Args:
            comm: The MPI communicator.
            rank: The rank of the current process.
            size: The total number of processes in the communicator.
            name: The name of the thread.
            q: The message queue to get messages to send.
        """
        super(MPISendThread, self).__init__()
        self._stop_event = threading.Event()
        self.comm = comm
        self.rank = rank
        self.size = size
        self.name = name
        self.q = q

    def run(self):
        """
        Run the MPI Send Thread.

        This method continuously checks the message queue and sends messages to the specified destination.
        """
        logging.debug("Starting " + self.name +
                      ". Process ID = " + str(self.rank))
        while True:
            try:
                if not self.q.empty():
                    msg = self.q.get()
                    dest_id = msg.get(Message.MSG_ARG_KEY_RECEIVER)
                    self.comm.send(msg.to_string(), dest=dest_id)
                else:
                    time.sleep(0.003)
            except Exception:
                traceback.print_exc()
                raise Exception("MPI failed!")

    def stop(self):
        """
        Stop the MPI Send Thread.
        """
        self._stop_event.set()

    def stopped(self):
        """
        Check if the MPI Send Thread is stopped.

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
        Raise an exception in the MPI Send Thread to stop it.
        """
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            thread_id, ctypes.py_object(SystemExit)
        )
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print("Exception raise failure")
