from fedml.core.common.singleton import Singleton
from .compute_cache_manager import ComputeCacheManager
from .business_models import LogsUploadModel, LogRequestModel


class LogsManager(Singleton):
    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return LogsManager()

    @staticmethod
    def save_logs(logs_json):
        if logs_json is None:
            return

        log_model = LogsUploadModel(logs_json)
        ComputeCacheManager.get_instance().set_redis_params()
        ComputeCacheManager.get_instance().store_cache(log_model)

    @staticmethod
    def get_logs(logs_request):
        log_req_model = LogRequestModel(logs_request)
        ComputeCacheManager.get_instance().set_redis_params()
        return ComputeCacheManager.get_instance().get_logs(
            log_req_model.run_id, edge_id=log_req_model.edgeId,
            page_num=log_req_model.page_num, page_size=log_req_model.page_size)

