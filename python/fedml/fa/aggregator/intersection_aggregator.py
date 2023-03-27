from typing import List, Tuple, Any

from fedml.fa.base_frame.server_aggregator import FAServerAggregator


def get_intersection_of_two_lists_keep_duplicates(list1, list2):
    """
    Keep duplicates in the intersection, e.g., list1=[1,2,3,2,3], list2=[2,3,2,3]. intersect(list1, list2) = [2,3,2,3]
    :param list1: first list
    :param list2: second list
    :return: intersection of the 2 lists
    """
    intersection = []
    for i in range(len(list1)):
        for j in range(len(list2) - 1, -1, -1):
            if list1[i] == list2[j]:
                intersection.append(list2[j])
                list2.remove(j)
    return intersection


def get_intersection_of_two_lists_remove_duplicates(list1, list2):
    """
    Remove duplicates in the intersection, e.g., list1=[1,2,3,2,3], list2=[2,3,2,3]. intersect(list1, list2) = [2,3]
    :param list1: first list
    :param list2: second list
    :return: intersection of the 2 lists
    """
    return list(set(list1) & set(list2))


class IntersectionAggregatorFA(FAServerAggregator):
    def __init__(self, args):
        super().__init__(args)
        self.set_server_data(server_data=[])

    def aggregate(self, local_submission_list: List[Tuple[float, Any]]):
        for i in range(0, len(local_submission_list)):
            _, local_submission = local_submission_list[i]
            if len(self.server_data) == 0:
                # no need to remove duplicates even in ``remove duplicate'' mode,
                # as the duplicates will be removed in later computation
                self.server_data = local_submission
            else:
                self.server_data = get_intersection_of_two_lists_remove_duplicates(self.server_data, local_submission)
        print(f"cardinality = {self.get_cardinality()}")
        return self.server_data

    def get_cardinality(self):
        return len(self.server_data)
