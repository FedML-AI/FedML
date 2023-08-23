import json
import redis
from fedml.cli.model_deployment.device_server_constants import ServerConstants
from .device_model_db import FedMLModelDatabase


class FedMLModelCache(object):
    FEDML_MODEL_DEPLOYMENT_RESULT_TAG = "FEDML_MODEL_DEPLOYMENT_RESULT-"
    FEDML_MODEL_DEPLOYMENT_STATUS_TAG = "FEDML_MODEL_DEPLOYMENT_STATUS-"
    FEDML_MODEL_DEPLOYMENT_MONITOR_TAG = "FEDML_MODEL_DEPLOYMENT_MONITOR-"
    FEDML_MODEL_END_POINT_ACTIVATION_TAG = "FEDML_MODEL_END_POINT_ACTIVATION-"
    FEDML_MODEL_END_POINT_STATUS_TAG = "FEDML_MODEL_END_POINT_STATUS-"
    FEDML_MODEL_DEVICE_INFO_TAG = "FEDML_MODEL_DEVICE_INFO_TAG-"
    FEDML_MODEL_END_POINT_TOKEN_TAG = "FEDML_MODEL_END_POINT_TOKEN_TAG-"
    FEDML_MODEL_ROUND_ROBIN_PREVIOUS_DEVICE_TAG = "FEDML_MODEL_ROUND_ROBIN_PREVIOUS_DEVICE_TAG-"
    FEDML_KEY_COUNT_PER_SCAN = 1000

    def __init__(self):
        pass

    def __new__(cls, *args, **kw):
        if not hasattr(cls, "_instance"):
            orig = super(FedMLModelCache, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.redis_pool = None
        self.redis_connection = None
        self.model_deployment_db = FedMLModelDatabase().get_instance()
        self.model_deployment_db.create_table()

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
        return FedMLModelCache()

    def set_deployment_result(self, end_point_id, end_point_name,
                              model_name, model_version, device_id, deployment_result):
        result_dict = {"cache_device_id": device_id, "result": deployment_result}
        self.redis_connection.rpush(self.get_deployment_result_key(end_point_name, model_name), json.dumps(result_dict))
        self.model_deployment_db.set_deployment_result(end_point_id, end_point_name,
                                                       model_name, model_version,
                                                       device_id, deployment_result)

    def set_deployment_status(self, end_point_id, end_point_name,
                              model_name, model_version, device_id, deployment_status):
        status_dict = {"cache_device_id": device_id, "status": deployment_status}
        self.redis_connection.rpush(self.get_deployment_status_key(end_point_name, model_name), json.dumps(status_dict))
        self.model_deployment_db.set_deployment_status(end_point_id, end_point_name,
                                                       model_name, model_version,
                                                       device_id, deployment_status)

    def get_deployment_result_list(self, end_point_name, model_name):
        result_list = self.redis_connection.lrange(self.get_deployment_result_key(end_point_name, model_name), 0, -1)
        if result_list is None or len(result_list) <= 0:
            result_list = self.model_deployment_db.get_deployment_result_list(end_point_name, model_name)
            for result in result_list:
                self.redis_connection.rpush(self.get_deployment_result_key(end_point_name, model_name),
                                            json.dumps(result))
        return result_list

    def get_deployment_result_list_size(self, end_point_name, model_name):
        result_list = self.get_deployment_result_list(end_point_name, model_name)
        return len(result_list)

    def get_deployment_status_list(self, end_point_name, model_name):
        status_list = self.redis_connection.lrange(self.get_deployment_status_key(end_point_name, model_name), 0, -1)
        if status_list is None or len(status_list) <= 0:
            status_list = self.model_deployment_db.get_deployment_status_list(end_point_name, model_name)
            for status in status_list:
                self.redis_connection.rpush(self.get_deployment_status_key(end_point_name, model_name),
                                            json.dumps(status))
        return status_list

    def get_deployment_status_list_size(self, end_point_name, model_name):
        status_list = self.get_deployment_status_list(end_point_name, model_name)
        return len(status_list)

    def get_status_item_info(self, status_item):
        status_item_json = json.loads(status_item)
        device_id = status_item_json["cache_device_id"]
        status_payload = json.loads(status_item_json["status"])
        return device_id, status_payload

    def get_result_item_info(self, result_item):
        result_item_json = json.loads(result_item)
        device_id = result_item_json["cache_device_id"]
        result_payload = json.loads(result_item_json["result"])
        return device_id, result_payload

    def get_idle_device(self, end_point_name,
                        model_name, model_version,
                        check_end_point_status=True):
        # Find all deployed devices
        status_list = self.get_deployment_status_list(end_point_name, model_name)   # get from redis
        if len(status_list) == 0:
            return None, None

        idle_device_list = list()
        if model_version == "latest":
            _, status_payload = self.get_status_item_info(status_list[-1])
            model_version = status_payload["model_version"]

        # find all devices
        try:
            for status_item in status_list:
                device_id, status_payload = self.get_status_item_info(status_item)
                print(f"status_payload {status_payload}")
                model_status = status_payload["model_status"]
                model_version_cache = status_payload["model_version"]
                end_point_id_cache = status_payload["end_point_id"]
                print(f"model_version {model_version}, model_version_cache {model_version_cache}")
                if model_version == model_version_cache and \
                        model_status == ServerConstants.MSG_MODELOPS_DEPLOYMENT_STATUS_DEPLOYED:
                    idle_device_list.append({"device_id": device_id, "end_point_id": end_point_id_cache})
        except Exception as e:
            print("Get idel device list Failed:")
            print(e)
        print(f"idle device list {idle_device_list}")
        # Randomly shuffle 
        # shuffle the list of deployed devices and get the first one as the target idle device.
        # if len(idle_device_list) <= 0:
        #     return None, None
        # shuffle(idle_device_list)
        # idle_device_dict = idle_device_list[0]

        # Round Robin
        total_device_num = len(idle_device_list)
        redis_round_robin_key = self.get_round_robin_prev_device(end_point_name, model_name, model_version)

        try:
            if self.redis_connection.exists(redis_round_robin_key):
                selected_device_index = int(self.redis_connection.get(redis_round_robin_key))
                selected_device_index %= total_device_num
            else:
                selected_device_index = 0
            next_selected_device_index = (selected_device_index + 1) % total_device_num
            self.redis_connection.set(redis_round_robin_key, str(next_selected_device_index))
        except Exception as e:
            print("Inference Device selection Failed:")
            print(str(e))

        print(f"Using Round Robin, the device index is {selected_device_index}")
        idle_device_dict = idle_device_list[selected_device_index]
        # Find deployment result from the target idle device.
        try:
            result_list = self.get_deployment_result_list(end_point_name, model_name)
            for result_item in result_list:
                device_id, result_payload = self.get_result_item_info(result_item)
                found_end_point_id = result_payload["end_point_id"]
                found_end_point_name = result_payload["end_point_name"]
                # Check whether the end point is activated.
                if check_end_point_status:
                    end_point_activated = self.get_end_point_activation(found_end_point_id)
                    if not end_point_activated:
                        continue

                if found_end_point_id == idle_device_dict["end_point_id"] \
                        and device_id == idle_device_dict["device_id"]:
                    print(f"The chosen device is {device_id}")
                    return result_payload, device_id
        except Exception as e:
            print(e)

        return None, None

    def set_end_point_status(self, end_point_id, end_point_name, status):
        self.redis_connection.set(self.get_end_point_status_key(end_point_id), status)
        self.model_deployment_db.set_end_point_status(end_point_id, end_point_name, status)

    def get_end_point_status(self, end_point_id):
        if not self.redis_connection.exists(self.get_end_point_status_key(end_point_id)):
            status = self.model_deployment_db.get_end_point_status(end_point_id)
            if status is not None:
                self.redis_connection.set(self.get_end_point_status_key(end_point_id), status)
            return status

        status = self.redis_connection.get(self.get_end_point_status_key(end_point_id))
        return status

    def set_end_point_activation(self, end_point_id, end_point_name, activate_status):
        status = 1 if activate_status else 0
        self.redis_connection.set(self.get_end_point_activation_key(end_point_id), status)
        self.model_deployment_db.set_end_point_activation(end_point_id, end_point_name, status)
        
    def delete_end_point(self, end_point_name, model_name, model_version):
        self.redis_connection.delete(self.get_deployment_status_key(end_point_name, model_name))
        # TODO: Delete related KV Pair        

    def get_end_point_activation(self, end_point_id):
        if not self.redis_connection.exists(self.get_end_point_activation_key(end_point_id)):
            activated = self.model_deployment_db.get_end_point_activation(end_point_id)
            self.redis_connection.set(self.get_end_point_activation_key(end_point_id), activated)
            status = True if int(activated) == 1 else False
            return status

        status_int = self.redis_connection.get(self.get_end_point_activation_key(end_point_id))
        status = True if int(status_int) == 1 else False
        return status

    def set_end_point_device_info(self, end_point_id, end_point_name, device_info):
        self.redis_connection.set(self.get_deployment_device_info_key(end_point_id), device_info)
        self.model_deployment_db.set_end_point_device_info(end_point_id, end_point_name, device_info)

    def get_end_point_device_info(self, end_point_id):
        if not self.redis_connection.exists(self.get_deployment_device_info_key(end_point_id)):
            device_info = self.model_deployment_db.get_end_point_device_info(end_point_id)
            if device_info is not None:
                self.redis_connection.set(self.get_deployment_device_info_key(end_point_id), device_info)
            return device_info

        device_info = self.redis_connection.get(self.get_deployment_device_info_key(end_point_id))
        return device_info

    def set_end_point_token(self, end_point_id, end_point_name, model_name, token):
        if self.redis_connection.exists(self.get_deployment_token_key(end_point_name, model_name)):
            return
        self.redis_connection.set(self.get_deployment_token_key(end_point_name, model_name), token)
        self.model_deployment_db.set_end_point_token(end_point_id, end_point_name, model_name, token)

    def get_end_point_token(self, end_point_name, model_name):
        if not self.redis_connection.exists(self.get_deployment_token_key(end_point_name, model_name)):
            token = self.model_deployment_db.get_end_point_token(end_point_name, model_name)
            if token is not None:
                self.redis_connection.set(self.get_deployment_token_key(end_point_name, model_name), token)
            return token

        token = self.redis_connection.get(self.get_deployment_token_key(end_point_name, model_name))
        return token

    def get_deployment_result_key(self, end_point_name, model_name):
        return "{}{}-{}".format(FedMLModelCache.FEDML_MODEL_DEPLOYMENT_RESULT_TAG, end_point_name, model_name)

    def get_deployment_status_key(self, end_point_name, model_name):
        return "{}{}-{}".format(FedMLModelCache.FEDML_MODEL_DEPLOYMENT_STATUS_TAG, end_point_name, model_name)

    def get_end_point_status_key(self, end_point_id):
        return "{}{}".format(FedMLModelCache.FEDML_MODEL_END_POINT_STATUS_TAG, end_point_id)

    def get_end_point_activation_key(self, end_point_id):
        return "{}{}".format(FedMLModelCache.FEDML_MODEL_END_POINT_ACTIVATION_TAG, end_point_id)

    def get_deployment_device_info_key(self, end_point_id):
        return "{}{}".format(FedMLModelCache.FEDML_MODEL_DEVICE_INFO_TAG, end_point_id)

    def get_deployment_token_key(self, end_point_name, model_name):
        return "{}{}-{}".format(FedMLModelCache.FEDML_MODEL_END_POINT_TOKEN_TAG, end_point_name, model_name)

    def get_round_robin_prev_device(self, end_point_name, model_name, version):
        return "{}{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_ROUND_ROBIN_PREVIOUS_DEVICE_TAG, end_point_name, model_name, version)

    def set_monitor_metrics(self, end_point_id, end_point_name,
                            model_name, model_version,
                            total_latency, avg_latency,
                            total_request_num, current_qps,
                            avg_qps, timestamp, device_id):
        metrics_dict = {"total_latency": total_latency, "avg_latency": avg_latency,
                        "total_request_num": total_request_num, "current_qps": current_qps,
                        "avg_qps": avg_qps, "timestamp": timestamp, "device_id": device_id}
        self.redis_connection.rpush(self.get_monitor_metrics_key(end_point_name, model_name, model_version),
                                    json.dumps(metrics_dict))
        self.model_deployment_db.set_monitor_metrics(end_point_id, end_point_name,
                                                     model_name, model_version,
                                                     total_latency, avg_latency,
                                                     total_request_num, current_qps,
                                                     avg_qps, timestamp, device_id)

    def get_latest_monitor_metrics(self, end_point_name, model_name, model_version):
        if not self.redis_connection.exists(self.get_monitor_metrics_key(end_point_name, model_name, model_version)):
            metrics_dict = self.model_deployment_db.get_latest_monitor_metrics(end_point_name, model_name, model_version)
            if metrics_dict is not None:
                self.redis_connection.rpush(self.get_monitor_metrics_key(end_point_name, model_name, model_version),
                                            metrics_dict)
            return metrics_dict

        return self.redis_connection.lindex(self.get_monitor_metrics_key(end_point_name, model_name, model_version), -1)

    def get_monitor_metrics_item(self, end_point_name, model_name, model_version, index):
        if not self.redis_connection.exists(self.get_monitor_metrics_key(end_point_name, model_name, model_version)):
            metrics_dict = self.model_deployment_db.get_monitor_metrics_item(end_point_name, model_name, model_version, index)
            if metrics_dict is not None:
                self.redis_connection.rpush(self.get_monitor_metrics_key(end_point_name, model_name, model_version),
                                            metrics_dict)
                return metrics_dict, index+1
            return None, 0

        metrics_item = self.redis_connection.lindex(self.get_monitor_metrics_key(end_point_name, model_name,
                                                                                 model_version), index)
        return metrics_item, index+1

    def get_metrics_item_info(self, metrics_item):
        metrics_item_json = json.loads(metrics_item)
        total_latency = metrics_item_json["total_latency"]
        avg_latency = metrics_item_json["avg_latency"]
        total_request_num = metrics_item_json["total_request_num"]
        current_qps = metrics_item_json["current_qps"]
        avg_qps = metrics_item_json["avg_qps"]
        timestamp = metrics_item_json["timestamp"]
        device_id = metrics_item_json["device_id"]
        return total_latency, avg_latency, total_request_num, current_qps, avg_qps, timestamp, device_id

    def get_monitor_metrics_key(self, end_point_name, model_name, model_version):
        return "{}{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_DEPLOYMENT_MONITOR_TAG,
                                   end_point_name, model_name, model_version)


if __name__ == "__main__":
    _end_point_id_ = "4f63aa70-312e-4a9c-872d-cc6e8d95f95b"
    _status_list_ = FedMLModelCache.get_instance().get_deployment_status_list(_end_point_id_)
    _result_list_ = FedMLModelCache.get_instance().get_deployment_result_list(_end_point_id_)
    idle_result_payload = FedMLModelCache.get_instance().get_idle_device(_end_point_id_)
