# -*- coding: utf-8 -*-n
__all__ = ['ENV', 'OUT_DIR_PATH']

from conf.conf import *
from conf.env import EnvWrapper

ENV = EnvWrapper(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), True, 'TrainingExecutor')
