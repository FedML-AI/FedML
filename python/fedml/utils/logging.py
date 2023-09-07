import logging

#define log levels
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
        """
        Create and configure a logger.

        Args:
            name (str): The name of the logger.
            level: The logging level for the logger.

        Returns:
            logger: An instance of the logger.
        """
        if name is None:
            raise ValueError("name for logger cannot be None")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        return logger_


logger = LoggerCreator.create_logger(name="FedML", level=logging.INFO)
