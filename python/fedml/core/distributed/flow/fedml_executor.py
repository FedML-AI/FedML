import abc


class FedMLExecutor(abc.ABC):
    """
    Abstract base class for Federated Machine Learning Executors.

    This class defines the basic structure and methods for a FedML executor.

    Args:
        id (str): Identifier for the executor.
        neighbor_id_list (List[str]): List of neighbor executor IDs.

    Attributes:
        id (str): Identifier for the executor.
        neighbor_id_list (List[str]): List of neighbor executor IDs.
        params (Any): Parameters associated with the executor.
        context (Any): Context or environment information.

    Methods:
        get_context() -> Any:
        Get the context or environment information associated with the executor.

        set_context(context: Any) -> None:
        Set the context or environment information for the executor.

        get_params() -> Any:
        Get the parameters associated with the executor.

        set_params(params: Any) -> None:
        Set the parameters for the executor.

        set_id(id: str) -> None:
        Set the identifier for the executor.

        set_neighbor_id_list(neighbor_id_list: List[str]) -> None:
        Set the list of neighbor executor IDs.

        get_id() -> str:
        Get the identifier of the executor.

        get_neighbor_id_list() -> List[str]:
        Get the list of neighbor executor IDs.
    """

    def __init__(self, id, neighbor_id_list):
        """
        Initialize a FedMLExecutor.

        Args:
            id (str): Identifier for the executor.
            neighbor_id_list (List[str]): List of neighbor executor IDs.
        """
        self.id = id
        self.neighbor_id_list = neighbor_id_list
        self.params = None
        self.context = None

    def get_context(self) -> Any:
        """
        Get the context or environment information associated with the executor.

        Returns:
            Any: The context or environment information.
        """
        return self.context

    def set_context(self, context: Any) -> None:
        """
        Set the context or environment information for the executor.

        Args:
            context (Any): The context or environment information.
        """
        self.context = context

    def get_params(self) -> Any:
        """
        Get the parameters associated with the executor.

        Returns:
            Any: The parameters.
        """
        return self.params

    def set_params(self, params: Any) -> None:
        """
        Set the parameters for the executor.

        Args:
            params (Any): The parameters.
        """
        self.params = params

    def set_id(self, id: str) -> None:
        """
        Set the identifier for the executor.

        Args:
            id (str): The identifier.
        """
        self.id = id

    def set_neighbor_id_list(self, neighbor_id_list: List[str]) -> None:
        """
        Set the list of neighbor executor IDs.

        Args:
            neighbor_id_list (List[str]): List of neighbor executor IDs.
        """
        self.neighbor_id_list = neighbor_id_list

    def get_id(self) -> str:
        """
        Get the identifier of the executor.

        Returns:
            str: The identifier.
        """
        return self.id

    def get_neighbor_id_list(self) -> List[str]:
        """
        Get the list of neighbor executor IDs.

        Returns:
            List[str]: List of neighbor executor IDs.
        """
        return self.neighbor_id_list
