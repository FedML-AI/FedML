
from fedml.core.common.singleton import Singleton


class FedMLLaunchDataInterface(Singleton):
    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLLaunchDataInterface()


class FedMLLaunchDataInterface(object):

    def __init__(self):
        self.job_id = 0
        self.app_name = 0
        self.model_name = ""

    def show(self):
        pass
