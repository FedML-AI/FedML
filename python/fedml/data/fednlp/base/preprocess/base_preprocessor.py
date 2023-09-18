from abc import ABC, abstractmethod


from abc import ABC, abstractmethod

class BasePreprocessor(ABC):
    """Abstract base class for data preprocessors.

    This class defines the common interface for data preprocessors, which are responsible for transforming
    and preparing data for further processing or analysis.

    Attributes:
        **kwargs: Additional keyword arguments specific to the preprocessor implementation.

    Methods:
        transform(*args): Abstract method to transform data.

    """

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the BasePreprocessor with optional keyword arguments.

        Args:
            **kwargs: Additional keyword arguments specific to the preprocessor implementation.
        """
        self.__dict__.update(kwargs)

    @abstractmethod
    def transform(self, *args):
        """Transform data using the preprocessor.

        This method should be implemented by subclasses to apply data transformation operations.

        Args:
            *args: Variable-length arguments representing the input data to be transformed.

        Returns:
            Transformed data or processed result.
        """
        pass
