from typing import List, Tuple, Any

from fedml.fa.base_frame.server_aggregator import FAServerAggregator


def get_intersection_of_two_lists_keep_duplicates(list1: List[Any], list2: List[Any]) -> List[Any]:
    """
    Return the intersection of two lists while keeping duplicates.

    Args:
        list1 (List): The first list.
        list2 (List): The second list.

    Returns:
        List: The intersection of the two lists, keeping duplicates.
    """
    intersection = []
    for item in list1:
        if item in list2:
            intersection.append(item)
            list2.remove(item)
    return intersection


def get_intersection_of_two_lists_remove_duplicates(list1: List[Any], list2: List[Any]) -> List[Any]:
    """
    Return the intersection of two lists and remove duplicate values.

    Args:
        list1 (List): The first list.
        list2 (List): The second list.

    Returns:
        List: The intersection of the two lists with duplicates removed.
    """
    return list(set(list1) & set(list2))


class IntersectionAggregatorFA(FAServerAggregator):
    def __init__(self, args):
        """
        Initialize the IntersectionAggregatorFA.

        Args:
            args: Additional arguments for initialization.

        Returns:
            None
        """
        super().__init__(args)
        self.set_server_data(server_data=[])

    def aggregate(self, local_submission_list: List[Tuple[float, Any]]) -> List[Any]:
        """
        Aggregate local submissions while maintaining intersection.

        Args:
            local_submission_list (List[Tuple[float, Any]]): A list of local submissions.

        Returns:
            List: The intersection of local submissions.
        """
        for _, local_submission in local_submission_list:
            if len(self.server_data) == 0:

                self.server_data = local_submission
            else:
                self.server_data = get_intersection_of_two_lists_remove_duplicates(self.server_data, local_submission)
        print(f"cardinality = {self.get_cardinality()}")
        return self.server_data

    def get_cardinality(self) -> int:
        """
        Get the cardinality (number of elements) of the aggregated data.

        Returns:
            int: The cardinality of the aggregated data.
        """
        return len(self.server_data)
