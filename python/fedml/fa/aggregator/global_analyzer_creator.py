from fedml.fa.aggregator.avg_aggregator import AVGAggregatorFA
from fedml.fa.aggregator.frequency_estimation_aggregator import FrequencyEstimationAggregatorFA
from fedml.fa.aggregator.heavy_hitter_triehh_aggregator import HeavyHitterTriehhAggregatorFA
from fedml.fa.aggregator.intersection_aggregator import IntersectionAggregatorFA
from fedml.fa.aggregator.k_percentile_element_aggregator import KPercentileElementAggregatorFA
from fedml.fa.aggregator.union_aggregator import UnionAggregatorFA
from fedml.fa.constants import FA_TASK_AVG, FA_TASK_INTERSECTION, FA_TASK_CARDINALITY, FA_TASK_HISTOGRAM, FA_TASK_FREQ, \
    FA_TASK_UNION, FA_TASK_K_PERCENTILE_ELEMENT, FA_TASK_HEAVY_HITTER_TRIEHH


def create_global_analyzer(args, train_data_num):
    task_type = args.fa_task
    if task_type == FA_TASK_AVG:
        return AVGAggregatorFA(args)
    if task_type == FA_TASK_INTERSECTION or task_type == FA_TASK_CARDINALITY:
        return IntersectionAggregatorFA(args)
    if task_type == FA_TASK_FREQ or task_type == FA_TASK_HISTOGRAM:
        return FrequencyEstimationAggregatorFA(args)
    if task_type == FA_TASK_UNION:
        return UnionAggregatorFA(args)
    if task_type == FA_TASK_K_PERCENTILE_ELEMENT:
        return KPercentileElementAggregatorFA(args, train_data_num)
    if task_type == FA_TASK_HEAVY_HITTER_TRIEHH:
        return HeavyHitterTriehhAggregatorFA(args, train_data_num)



