import threading


class Singleton(object):
    _instance = None
    # For thread safety
    _lock = threading.Lock()

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            with cls._lock:
                # Another thread could have created the instance before we acquired the lock. So check that the
                # instance is still nonexistent.
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
