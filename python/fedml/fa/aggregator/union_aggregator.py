from typing import List, Tuple, Any
from fedml.fa.base_frame.server_aggregator import FAServerAggregator


def get_union_of_two_lists_keep_duplicates(list1, list2):
    """
    Compute the union of two lists while keeping duplicates.

    Args:
        list1 (List): The first list.
        list2 (List): The second list.

    Returns:
        List: The union of the two lists with duplicates.
    """
    union = []
    for item in list1:
        union.append(item)
        if item in list2:
            list2.remove(item)
    union.extend(list2)
    return union


def get_union_of_two_lists_remove_duplicates(list1, list2):
    """
    Compute the union of two lists and remove duplicates.

    Args:
        list1 (List): The first list.
        list2 (List): The second list.

    Returns:
        List: The union of the two lists without duplicates.
    """
    return list(set(list1 + list2))


class UnionAggregatorFA(FAServerAggregator):
    def __init__(self, args):
        """
        Initialize the UnionAggregatorFA.

        Args:
            args: Configuration arguments.

        Returns:
            None
        """
        super().__init__(args)
        self.set_server_data(server_data=[])
        self.union_function = get_union_of_two_lists_remove_duplicates  # Select the way to compute union

    def aggregate(self, local_submission_list: List[Tuple[float, Any]]):
        """
        Aggregate local submissions from clients.

        Args:
            local_submission_list (List[Tuple[float, Any]]): A list of tuples containing local submissions and weights.

        Returns:
            List: The aggregated result.
        """
        for i in range(0, len(local_submission_list)):
            _, local_submission = local_submission_list[i]
            # When server_data is [], i.e., the first round, will only process local_submission
            self.server_data = self.union_function(self.server_data, local_submission)
        return self.server_data
