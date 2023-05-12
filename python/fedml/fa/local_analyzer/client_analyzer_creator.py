from fedml.fa.constants import FA_TASK_AVG, FA_TASK_HEAVY_HITTER_TRIEHH, FA_TASK_UNION, FA_TASK_K_PERCENTILE_ELEMENT, \
    FA_TASK_INTERSECTION, FA_TASK_CARDINALITY, FA_TASK_HISTOGRAM, FA_TASK_FREQ
from fedml.fa.local_analyzer.avg import AverageClientAnalyzer
from fedml.fa.local_analyzer.frequency_estimation import FrequencyEstimationClientAnalyzer
from fedml.fa.local_analyzer.heavy_hitter_triehh import TrieHHClientAnalyzer
from fedml.fa.local_analyzer.intersection import IntersectionClientAnalyzer
from fedml.fa.local_analyzer.k_percentage_element import KPercentileElementClientAnalyzer
from fedml.fa.local_analyzer.union import UnionClientAnalyzer


def create_local_analyzer(args):
    task_type = args.fa_task
    if task_type == FA_TASK_AVG:
        return AverageClientAnalyzer(args)
    # if task_type == FA_TASK_HEAVY_HITTER_TRIEHH:
    #     return TrieHHClientAnalyzer(args)
    if task_type == FA_TASK_UNION:
        return UnionClientAnalyzer(args)
    if task_type == FA_TASK_K_PERCENTILE_ELEMENT:
        return KPercentileElementClientAnalyzer(args)
    if task_type == FA_TASK_INTERSECTION or task_type == FA_TASK_CARDINALITY:
        return IntersectionClientAnalyzer(args)
    if task_type == FA_TASK_FREQ or task_type == FA_TASK_HISTOGRAM:
        return FrequencyEstimationClientAnalyzer(args)
    if task_type == FA_TASK_HEAVY_HITTER_TRIEHH:
        return TrieHHClientAnalyzer(args)

