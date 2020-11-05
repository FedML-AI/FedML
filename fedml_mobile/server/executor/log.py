# -*- coding: utf-8 -*-
import sys
import logging
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from logging.handlers import TimedRotatingFileHandler
from fedml_mobile.server.executor.conf import ENV

LOG_FORMAT = "%(asctime)s %(levelname)s %(filename)s:%(lineno)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"


class LoggerWrapper:
    def __init__(self):
        self._console_init = False
        self.__logger = self._gen_logger(ENV.current_task_name, 'fedml')
        self.debug = self.__logger.debug
        self.info = self.__logger.info
        self.warning = self.__logger.warning
        self.error = self.__logger.error
        self.critical = self.__logger.critical
        self.exception = self.__logger.exception

    @staticmethod
    def _get_path(path):
        """
        :param path:
        :return:
        """
        if path != 'logs':
            path = os.path.join('logs', path)

        path = os.path.join(ENV.module_path, path)
        if not os.path.exists(path):
            os.makedirs(path)

        return path

    def _gen_logger(self, action='logs', log_name='Training'):
        base_logger = logging.getLogger(log_name)
        base_logger.setLevel(logging.DEBUG)

        log_file = os.path.join(self._get_path(action), log_name + ".log")
        ch = TimedRotatingFileHandler(log_file, when='D', encoding="utf-8")
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        ch.setFormatter(formatter)
        base_logger.addHandler(ch)
        base_logger.propagate = 0

        if ENV.console_log_enable and not self._console_init:
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            console.setFormatter(formatter)
            base_logger.addHandler(console)
            self._console_init = True

        return base_logger


__log = LoggerWrapper()
debug = __log.debug
info = __log.info
error = __log.error
warn = __log.warning
exception = __log.exception
critical = __log.critical
