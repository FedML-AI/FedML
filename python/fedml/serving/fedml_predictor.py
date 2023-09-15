import platform
import os
import stat
import logging
import traceback
from abc import ABC, abstractmethod
from ..computing.scheduler.model_scheduler.device_client_constants import ClientConstants
from ..computing.scheduler.comm_utils import sys_utils

class FedMLPredictor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass