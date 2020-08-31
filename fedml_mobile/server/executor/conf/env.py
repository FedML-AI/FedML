# -*- coding: utf-8 -*-


class EnvWrapper:
    def __init__(self, module_dir=None, log_enable=False, task_name='TrainingExecutor'):
        self._module_path = module_dir
        self._log_enable = log_enable
        self._current_task_name = task_name

    @property
    def module_path(self):
        return self._module_path

    @property
    def console_log_enable(self):
        return self._log_enable

    @console_log_enable.setter
    def console_log_enable(self, enable):
        self._log_enable = enable

    @property
    def current_task_name(self):
        return self._current_task_name
