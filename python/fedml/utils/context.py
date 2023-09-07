import threading
import traceback
from contextlib import contextmanager

from mpi4py import MPI


@contextmanager
def raise_MPI_error():
    """
    Context manager to catch and handle MPI-related errors.

    This context manager is used to catch exceptions and errors that may occur
    during MPI (Message Passing Interface) operations and handle them gracefully.

    Usage:
    ```python
    with raise_MPI_error():
        # Code that may raise MPI-related errors
    ```

    Returns:
        None
    """
    import logging

    logging.debug("Debugging, Enter the MPI catch error")
    try:
        yield
    except Exception as e:
        logging.info(e)
        logging.info("traceback.format_exc():\n%s" % traceback.format_exc())
        MPI.COMM_WORLD.Abort()


@contextmanager
def raise_error_without_process():
    """
    Context manager to catch and handle errors without aborting the MPI process.

    This context manager is used to catch exceptions and errors without aborting
    the MPI (Message Passing Interface) process, allowing it to continue running.

    Usage:
    ```python
    with raise_error_without_process():
        # Code that may raise errors
    ```

    Returns:
        None
    """
    import logging

    logging.debug("Debugging, Enter the MPI catch error")
    try:
        yield
    except Exception as e:
        logging.info(e)
        logging.info("traceback.format_exc():\n%s" % traceback.format_exc())


@contextmanager
def get_lock(lock: threading.Lock()):
    """
    Context manager to acquire and release a threading lock.

    This context manager is used to acquire and release a threading lock in a controlled
    manner. It ensures that the lock is always released, even in the presence of exceptions.

    Args:
        lock (threading.Lock): The threading lock to acquire and release.

    Usage:
    ```python
    my_lock = threading.Lock()
    with get_lock(my_lock):
        # Code that requires the lock
    # The lock is automatically released after the code block
    ```

    Returns:
        None
    """
    lock.acquire()
    yield
    if lock.locked():
        lock.release()
