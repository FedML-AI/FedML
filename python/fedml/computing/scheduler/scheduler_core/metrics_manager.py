from fedml.core.common.singleton import Singleton
from .compute_cache_manager import ComputeCacheManager
from .business_models import MetricsModel


class MetricsManager(Singleton):
    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return MetricsManager()

    def save_metrics(self, metrics_json):
        try:
            if metrics_json is None:
                return

            metrics_model = MetricsModel(metrics_json)
            ComputeCacheManager.get_instance().set_redis_params()
            ComputeCacheManager.get_instance().store_cache(metrics_model)
        except Exception as e:
            pass

    def get_metrics(self):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            return ComputeCacheManager.get_instance().get_metrics()
        except Exception as e:
            return None

