# -*- coding: utf-8 -*-n
import os
from fedml_mobile.server.executor.conf.env import EnvWrapper
ENV = EnvWrapper(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), True, 'TrainingExecutor')
