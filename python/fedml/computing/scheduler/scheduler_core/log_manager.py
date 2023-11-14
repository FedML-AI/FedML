from fedml.core.common.singleton import Singleton
from .compute_cache_manager import ComputeCacheManager
from .business_models import LogsUploadModel, LogRequestModel, LogResponseModel


class LogsManager(Singleton):
    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return LogsManager()

    def save_logs(self, logs_json):
        if logs_json is None:
            return

        log_model = LogsUploadModel(logs_json)
        ComputeCacheManager.get_instance().set_redis_params()
        ComputeCacheManager.get_instance().store_cache(log_model)

    def get_logs(self, run_id, edge_id=-1, page_num=1, page_size=100):
        ComputeCacheManager.get_instance().set_redis_params()
        response_model = ComputeCacheManager.get_instance().get_logs(
            run_id, edge_id=edge_id, page_num=page_num, page_size=page_size)
        return response_model

    def get_logs(self, logs_request):
        log_req_model = LogRequestModel(logs_request)
        ComputeCacheManager.get_instance().set_redis_params()
        return ComputeCacheManager.get_instance().get_logs(
            log_req_model.run_id, edge_id=log_req_model.edgeId,
            page_num=log_req_model.page_num, page_size=log_req_model.page_size)

