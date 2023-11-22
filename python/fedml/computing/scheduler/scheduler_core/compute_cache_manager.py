import redis
from .compute_gpu_cache import ComputeGpuCache
from .compute_logs_cache import ComputeLogsCache
from .business_models import LogsUploadModel, MetricsModel


class ComputeCacheManager(object):

    def __init__(self):
        pass

    def __new__(cls, *args, **kw):
        if not hasattr(cls, "_instance"):
            orig = super(ComputeCacheManager, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.redis_pool = None
        self.redis_connection = None
        self.gpu_cache = ComputeGpuCache(self.redis_connection)
        self.logs_cache = ComputeLogsCache(self.redis_connection)

    def setup_redis_connection(self, redis_addr, redis_port, redis_password="fedml_default"):
        if redis_password is None or redis_password == "" or redis_password == "fedml_default":
            self.redis_pool = redis.ConnectionPool(host=redis_addr, port=int(redis_port), decode_responses=True)
        else:
            self.redis_pool = redis.ConnectionPool(host=redis_addr, port=int(redis_port),
                                                   password=redis_password, decode_responses=True)
        self.redis_connection = redis.Redis(connection_pool=self.redis_pool)
        self.gpu_cache.redis_connection = self.redis_connection
        self.logs_cache.redis_connection = self.redis_connection

    def set_redis_params(self, redis_addr="local", redis_port=6379, redis_password="fedml_default"):
        if self.redis_pool is None:
            if redis_addr is None or redis_addr == "local":
                self.setup_redis_connection("localhost", redis_port, redis_password)
            else:
                self.setup_redis_connection(redis_addr, redis_port, redis_password)

    def get_redis_connection(self):
        return self.redis_connection

    @staticmethod
    def get_instance(redis_addr="local", redis_port=6379):
        return ComputeCacheManager()

    def get_gpu_cache(self):
        return self.gpu_cache

    def store_cache(self, cache_data):
        if isinstance(cache_data, LogsUploadModel):
            self.logs_cache.save_run_logs(cache_data)
        elif isinstance(cache_data, MetricsModel):
            pass
        pass

    def store_metrics(self):
        pass

    def store_model(self):
        pass

    def store_artifact_logs(self):
        pass

    def store_artifacts(self):
        pass

    def get_metrics(self):
        pass

    def get_logs(self, run_id, edge_id=-1, page_num=1, page_size=100):
        return self.logs_cache.get_run_logs(run_id, edge_id=edge_id, page_num=page_num, page_size=page_size)

    def get_model(self):
        pass

    def get_artifact_logs(self):
        pass

    def get_artifacts(self):
        pass





