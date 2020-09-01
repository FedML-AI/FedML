# -*- coding: utf-8 -*-n
__all__ = ['ENV', 'OUT_DIR_PATH', 'MQTT_BROKER_HOST', 'MQTT_BROKER_PORT', 'MODEL_FOLDER_PATH', 'RESOURCE_DIR_PATH']
import os
from fedml_mobile.server.executor.conf.conf import *
from fedml_mobile.server.executor.conf.env import EnvWrapper
ENV = EnvWrapper(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), True, 'TrainingExecutor')
