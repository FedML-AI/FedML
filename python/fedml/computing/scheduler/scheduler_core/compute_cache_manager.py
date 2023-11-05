from fedml.core.common.singleton import Singleton
import redis

class ComputeCacheManager(object):

    FEDML_GLOBAL_DEVICE_RUN_NUM_GPUS_TAG = "FEDML_GLOBAL_DEVICE_RUN_NUM_GPUS_TAG-"
    FEDML_GLOBAL_DEVICE_RUN_GPU_IDS_TAG = "FEDML_GLOBAL_DEVICE_RUN_GPU_IDS_TAG-"
    FEDML_GLOBAL_DEVICE_AVAILABLE_GPU_IDS_TAG = "FEDML_GLOBAL_DEVICE_AVAILABLE_GPU_IDS_TAG-"
    FEDML_GLOBAL_DEVICE_TOTAL_NUM_GPUS_TAG = "FEDML_GLOBAL_DEVICE_TOTAL_NUM_GPUS_TAG-"
    FEDML_GLOBAL_RUN_TOTAL_NUM_GPUS_TAG = "FEDML_GLOBAL_RUN_TOTAL_NUM_GPUS_TAG-"
    FEDML_GLOBAL_RUN_DEVICE_IDS_TAG = "FEDML_GLOBAL_RUN_DEVICE_IDS_TAG-"
    FEDML_GLOBAL_GPU_SCHEDULER_LOCK = "FEDML_GLOBAL_GPU_SCHEDULER_LOCK"
    FEDML_DEVICE_RUN_LOCK_TAG = "FEDML_DEVICE_RUN_LOCK_TAG-"
    FEDML_DEVICE_LOCK_TAG = "FEDML_DEVICE_LOCK_TAG-"
    FEDML_RUN_LOCK_TAG = "FEDML_RUN_LOCK_TAG-"

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

    def setup_redis_connection(self, redis_addr, redis_port, redis_password="fedml_default"):
        if redis_password is None or redis_password == "" or redis_password == "fedml_default":
            self.redis_pool = redis.ConnectionPool(host=redis_addr, port=int(redis_port), decode_responses=True)
        else:
            self.redis_pool = redis.ConnectionPool(host=redis_addr, port=int(redis_port),
                                                   password=redis_password, decode_responses=True)
        self.redis_connection = redis.Redis(connection_pool=self.redis_pool)

    def set_redis_params(self, redis_addr="local", redis_port=6379, redis_password="fedml_default"):
        if self.redis_pool is None:
            if redis_addr is None or redis_addr == "local":
                self.setup_redis_connection("localhost", redis_port, redis_password)
            else:
                self.setup_redis_connection(redis_addr, redis_port, redis_password)

    @staticmethod
    def get_instance(redis_addr="local", redis_port=6379):
        return ComputeCacheManager()

    def store_cache(self):
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

    def get_model(self):
        pass

    def get_artifact_logs(self):
        pass

    def get_artifacts(self):
        pass

    def get_redis_connection(self):
        return self.redis_connection

    def get_device_run_num_gpus(self, device_id, run_id):
        if self.redis_connection.exists(self.get_device_run_num_gpus_key(device_id, run_id)):
            device_run_num_gpus = self.redis_connection.get(self.get_device_run_num_gpus_key(device_id, run_id))
        else:
            device_run_num_gpus = 0

        return device_run_num_gpus

    def get_device_run_gpu_ids(self, device_id, run_id):
        if self.redis_connection.exists(self.get_device_run_gpu_ids_key(device_id, run_id)):
            device_run_gpu_ids = self.redis_connection.get(self.get_device_run_gpu_ids_key(device_id, run_id))
            if str(device_run_gpu_ids).strip() == "":
                return None
            device_run_gpu_ids = self.map_str_list_to_int_list(device_run_gpu_ids.split(','))
        else:
            device_run_gpu_ids = None

        return device_run_gpu_ids

    def get_device_available_gpu_ids(self, device_id):
        if self.redis_connection.exists(self.get_device_available_gpu_ids_key(device_id)):
            device_available_gpu_ids = self.redis_connection.get(self.get_device_available_gpu_ids_key(device_id))
            if str(device_available_gpu_ids).strip() == "":
                return []
            device_available_gpu_ids = self.map_str_list_to_int_list(device_available_gpu_ids.split(','))
        else:
            device_available_gpu_ids = list()

        return device_available_gpu_ids

    def get_device_total_num_gpus(self, device_id):
        if self.redis_connection.exists(self.get_device_total_num_gpus_key(device_id)):
            device_total_num_gpus = self.redis_connection.get(self.get_device_total_num_gpus_key(device_id))
        else:
            device_total_num_gpus = 0

        return device_total_num_gpus

    def get_run_total_num_gpus(self, run_id):
        if self.redis_connection.exists(self.get_run_total_num_gpus_key(run_id)):
            run_total_num_gpus = self.redis_connection.get(self.get_run_total_num_gpus_key(run_id))
        else:
            run_total_num_gpus = 0

        return run_total_num_gpus

    def get_run_device_ids_key(self, run_id):
        if self.redis_connection.exists(self.get_run_device_ids_key(run_id)):
            run_device_ids = self.redis_connection.get(self.get_run_device_ids_key(run_id))
            if str(run_device_ids).strip() == "":
                return None
            run_device_ids = run_device_ids.split(',')
        else:
            run_device_ids = None

        return run_device_ids

    def set_device_run_num_gpus(self, device_id, run_id, num_gpus):
        self.redis_connection.set(self.get_device_run_num_gpus_key(device_id, run_id), num_gpus)

    def set_device_run_gpu_ids(self, device_id, run_id, gpu_ids):
        if gpu_ids is None:
            if self.redis_connection.exists(self.get_device_run_gpu_ids_key(device_id, run_id)):
                self.redis_connection.delete(self.get_device_run_gpu_ids_key(device_id, run_id))
            return

        str_gpu_ids = self.map_list_to_str(gpu_ids)
        self.redis_connection.set(self.get_device_run_gpu_ids_key(device_id, run_id), str_gpu_ids)

    def set_device_available_gpu_ids(self, device_id, gpu_ids):
        str_gpu_ids = self.map_list_to_str(gpu_ids)
        self.redis_connection.set(self.get_device_available_gpu_ids_key(device_id), str_gpu_ids)

    def set_device_total_num_gpus(self, device_id, num_gpus):
        self.redis_connection.set(self.get_device_total_num_gpus_key(device_id), num_gpus)

    def set_run_total_num_gpus(self, run_id, num_gpus):
        self.redis_connection.set(self.get_run_total_num_gpus_key(run_id), num_gpus)

    def set_run_device_ids(self, run_id, device_ids):
        str_device_ids = self.map_list_to_str(device_ids)
        self.redis_connection.set(self.get_run_device_ids_key(run_id), str_device_ids)

    def get_device_run_num_gpus_key(self, device_id, run_id):
        return f"{ComputeCacheManager.FEDML_GLOBAL_DEVICE_RUN_NUM_GPUS_TAG}{device_id}_{run_id}"

    def get_device_run_gpu_ids_key(self, device_id, run_id):
        return f"{ComputeCacheManager.FEDML_GLOBAL_DEVICE_RUN_GPU_IDS_TAG}{device_id}_{run_id}"

    def get_device_available_gpu_ids_key(self, device_id):
        return f"{ComputeCacheManager.FEDML_GLOBAL_DEVICE_AVAILABLE_GPU_IDS_TAG}{device_id}"

    def get_device_total_num_gpus_key(self, device_id):
        return f"{ComputeCacheManager.FEDML_GLOBAL_DEVICE_TOTAL_NUM_GPUS_TAG}{device_id}"

    def get_run_total_num_gpus_key(self, run_id):
        return f"{ComputeCacheManager.FEDML_GLOBAL_RUN_TOTAL_NUM_GPUS_TAG}{run_id}"

    def get_run_device_ids_key(self, run_id):
        return f"{ComputeCacheManager.FEDML_GLOBAL_RUN_DEVICE_IDS_TAG}{run_id}"

    def get_device_run_lock_key(self, device_id, run_id):
        return f"{ComputeCacheManager.FEDML_DEVICE_RUN_LOCK_TAG}_{device_id}_{run_id}"

    def get_device_lock_key(self, device_id):
        return f"{ComputeCacheManager.FEDML_DEVICE_LOCK_TAG}_{device_id}"

    def get_run_lock_key(self, run_id):
        return f"{ComputeCacheManager.FEDML_RUN_LOCK_TAG}_{run_id}"

    def map_list_to_str(self, list_obj):
        list_map = map(lambda x: str(x), list_obj[0:])
        list_str = ",".join(list_map)
        return list_str

    def map_str_list_to_int_list(self, list_obj):
        list_map = map(lambda x: int(x), list_obj[0:])
        return list(list_map)


