import logging

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LoggerCreator:
    @staticmethod
    def create_logger(name=None, level=logging.INFO, args=None):
        """create a logger

        Args:
            name (str): name of the logger
            level: level of logger

        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        return logger_


logger = LoggerCreator.create_logger(name="FedML", level=logging.INFO)
