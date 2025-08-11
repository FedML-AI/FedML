import multiprocess as multiprocessing
import platform


class FedMLSharedResourceManager:
    def __init__(self):
        pass

    def __new__(cls, *args, **kw):
        if not hasattr(cls, "_instance"):
            orig = super(FedMLSharedResourceManager, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.shared_process_manager = None

    @staticmethod
    def get_instance():
        return FedMLSharedResourceManager()

    def get_shared_process_manager(self):
        if self.shared_process_manager is None:
            self.shared_process_manager = multiprocessing.Manager()
        return self.shared_process_manager

    def set_shared_process_manager(self, process_manager):
        self.shared_process_manager = process_manager

    def get_queue(self):
        return self.get_shared_process_manager().Queue()

    def get_event(self):
        if platform.system() == "Windows":
            return self.get_shared_process_manager().Event()

        return multiprocessing.Event()
