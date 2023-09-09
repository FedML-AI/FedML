import logging
from typing import List, Union
import torch

class OptRepo:
    """Collects and provides information about the subclasses of torch.optim.Optimizer.

    This class allows you to access and retrieve information about different PyTorch
    optimizer classes.

    Attributes:
        repo (dict): A dictionary containing optimizer class names as keys and the
            corresponding optimizer classes as values.

    Methods:
        get_opt_names(): Returns a list of supported optimizer names.
        name2cls(name: str): Returns the optimizer class based on its name.
        supported_parameters(opt: Union[str, torch.optim.Optimizer]): Returns a list of
            __init__ function parameters of an optimizer.
    """

    repo = {x.__name__.lower(): x for x in torch.optim.Optimizer.__subclasses__()}

    @classmethod
    def get_opt_names(cls) -> List[str]:
        """Returns a list of supported optimizer names.

        Returns:
            List[str]: Names of optimizers.
        """
        cls._update_repo()
        res = list(cls.repo.keys())
        return res

    @classmethod
    def name2cls(cls, name: str) -> torch.optim.Optimizer:
        """Returns the optimizer class belonging to the name.

        Args:
            name (str): Name of the optimizer.

        Returns:
            torch.optim.Optimizer: The class corresponding to the name.

        Raises:
            KeyError: If the provided optimizer name is invalid.
        """
        try:
            return cls.repo[name.lower()]
        except KeyError as e:
            logging.error(f"Invalid optimizer: {name}!")
            logging.error(f"These optimizers are registered: {cls.get_opt_names()}")
            raise e

    @classmethod
    def supported_parameters(cls, opt: Union[str, torch.optim.Optimizer]) -> List[str]:
        """Returns a list of __init__ function parameters of an optimizer.

        Args:
            opt (Union[str, torch.optim.Optimizer]): The name or class of the optimizer.

        Returns:
            List[str]: The list of the parameters.
        """
        if isinstance(opt, str):
            opt_ = cls.name2cls(opt)
        else:
            opt_ = opt

        res = list(opt_.__init__.__code__.co_varnames)
        res.remove("defaults")
        res.remove("self")
        res.remove("params")
        return res

    @classmethod
    def _update_repo(cls):
        cls.repo = {x.__name__: x for x in torch.optim.Optimizer.__subclasses__()}
