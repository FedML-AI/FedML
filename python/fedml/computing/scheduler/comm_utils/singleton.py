import threading


class Singleton(type):

    """
    We are not initializing the singleton objects using the `fedml.core.common.singleton`,
    because that Singleton approach does not allow to pass arguments during initialization.
    In particular, the error that is raised with the previous approach is:
    `TypeError: object.__new__() takes exactly one argument (the type to instantiate)`
    """

    _instances = {}
    # For thread safety
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # Another thread might have created the instance before the lock was acquired.
                # So check again if the instance is already created.
                if cls not in cls._instances:
                    cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
