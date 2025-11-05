import logging

class Logger:
    """
    A simple logger class that wraps around Python's logging module
    """
    def __init__(self, name: str, logging_level: str = "info"):
        """
        Initialize the Logger
        :param name: The name of the logger
        :param logging_level: The logging level (e.g., "debug", "info", "warning", "error")
        :return: None
        """
        self.logger = logging.getLogger(name)
        level = getattr(logging, logging_level.upper(), logging.INFO)
        self.logger.setLevel(level)

        fmt: logging.Formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch: logging.StreamHandler = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)
        self.name: str = name

    def info(self, msg: str) -> None:
        """
        Log an info message
        :param msg: The message to log
        :return: None
        """
        self.logger.info(f'{self.name} - {msg}')

    def debug(self, msg: str) -> None:
        """
        Log a debug message
        :param msg: The message to log
        :return: None
        """
        self.logger.debug(f'{self.name} - {msg}')

    def warning(self, msg: str) -> None:
        """
        Log a warning message
        :param msg: The message to log
        :return: None
        """
        self.logger.warning(f'{self.name} - {msg}')

    def error(self, msg: str) -> None:
        """
        Log an error message
        :param msg: The message to log
        :return: None
        """
        self.logger.error(f'{self.name} - {msg}')
