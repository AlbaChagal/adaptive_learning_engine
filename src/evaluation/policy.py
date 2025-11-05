from src.misc.config import Config
from src.logging.logger import Logger


class Policy:
    """
    Base class for all evaluation
    1. Must implement compute, select, and update methods
    2. Provides logging functionality
    3. Initialized with configuration and debug flag
    4. Usage:
       class specific_policy(Policy):
           pass
    5. Methods:
       - compute(*args, **kwargs): Compute method to be implemented by subclasses
       - select(*args, **kwargs): Select method to be implemented by subclasses
       - update(*args, **kwargs): Update method to be implemented by subclasses
    6. Raises NotImplementedError for unimplemented methods
    7. Attributes:
       - config: Configuration object
       - logger: Logger object
    8. Example:
       see linucb_policy.py and weakest_skill_first_policy.py for examples
    9. Note: This is an abstract base class and should not be instantiated directly.
    """
    def __init__(self,
                 config: Config,
                 is_debug: bool = False):
        self.config: Config = config
        self.logger: Logger = Logger(self.__class__.__name__, logging_level="debug" if is_debug else "info")

    def compute(self, *args, **kwargs):
        raise NotImplementedError("Compute method not implemented.")

    def select(self, *args, **kwargs):
        raise NotImplementedError("Select method not implemented.")

    def update(self, *args, **kwargs):
        raise NotImplementedError("Update method not implemented.")