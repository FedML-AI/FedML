import threading
import traceback
from contextlib import contextmanager

import mpi4py


@contextmanager
def raise_MPI_error():
    import logging

    logging.debug("Debugging, Enter the MPI catch error")
    try:
        yield
    except Exception as e:
        logging.info(e)
        logging.info("traceback.format_exc():\n%s" % traceback.format_exc())
        mpi4py.MPI.COMM_WORLD.Abort()


@contextmanager
def raise_error_without_process():
    import logging

    logging.debug("Debugging, Enter the MPI catch error")
    try:
        yield
    except Exception as e:
        logging.info(e)
        logging.info("traceback.format_exc():\n%s" % traceback.format_exc())


@contextmanager
def get_lock(lock: threading.Lock()):
    lock.acquire()
    yield
    if lock.locked():
        lock.release()
