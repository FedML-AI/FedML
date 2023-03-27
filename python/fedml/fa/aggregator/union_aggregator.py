from typing import List, Tuple, Any
from fedml.fa.base_frame.server_aggregator import FAServerAggregator


def get_union_of_two_lists_keep_duplicates(list1, list2):
    """
    Keep duplicates in the union, e.g., list1=[1,2,3,2,3], list2=[2,3,2,3]. intersect(list1, list2) = [1,2,3,2,3]
    :param list1: first list
    :param list2: second list
    :return: intersection of the 2 lists
    """
    union = []
    for item in list1:
        union.append(item)
        if item in list2:
            list2.remove(list2.index(item))
    union.extend(list2)
    return union


def get_union_of_two_lists_remove_duplicates(list1, list2):
    """
    Remove duplicates in the union, e.g., list1=[1,2,3,2,3], list2=[2,3,2,3]. intersect(list1, list2) = [1,2,3]
    :param list1: first list
    :param list2: second list
    :return: intersection of the 2 lists
    """
    return list(set(list1 + list2))


class UnionAggregatorFA(FAServerAggregator):
    def __init__(self, args):
        super().__init__(args)
        self.set_server_data(server_data=[])
        self.union_function = get_union_of_two_lists_remove_duplicates  # select the way to compute union

    def aggregate(self, local_submission_list: List[Tuple[float, Any]]):
        for i in range(0, len(local_submission_list)):
            _, local_submission = local_submission_list[i]
            # when server_data is [], i.e., the first round, will only process local_submission
            self.server_data = self.union_function(self.server_data, local_submission)
        return self.server_data
