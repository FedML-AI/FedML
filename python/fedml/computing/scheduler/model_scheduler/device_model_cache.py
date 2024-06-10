import json
import logging

import redis

from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
from .device_model_db import FedMLModelDatabase
from fedml.core.common.singleton import Singleton
from typing import Any, Dict, List
from fedml.computing.scheduler.scheduler_core.compute_gpu_cache import ComputeGpuCache


class FedMLModelCache(Singleton):
    FEDML_MODEL_DEPLOYMENT_RESULT_TAG = "FEDML_MODEL_DEPLOYMENT_RESULT-"
    FEDML_MODEL_DEPLOYMENT_STATUS_TAG = "FEDML_MODEL_DEPLOYMENT_STATUS-"
    FEDML_MODEL_DEPLOYMENT_MONITOR_TAG = "FEDML_MODEL_DEPLOYMENT_MONITOR-"
    FEDML_MODEL_END_POINT_ACTIVATION_TAG = "FEDML_MODEL_END_POINT_ACTIVATION-"
    FEDML_MODEL_END_POINT_STATUS_TAG = "FEDML_MODEL_END_POINT_STATUS-"
    FEDML_MODEL_DEVICE_INFO_TAG = "FEDML_MODEL_DEVICE_INFO_TAG-"
    FEDML_MODEL_END_POINT_TOKEN_TAG = "FEDML_MODEL_END_POINT_TOKEN_TAG-"
    FEDML_MODEL_ROUND_ROBIN_PREVIOUS_DEVICE_TAG = "FEDML_MODEL_ROUND_ROBIN_PREVIOUS_DEVICE_TAG-"
    FEDML_MODEL_ENDPOINT_REPLICA_NUM_TAG = "FEDML_MODEL_ENDPOINT_REPLICA_NUM_TAG-"

    # For scale-out & scale-in
    FEDML_MODEL_ENDPOINT_REPLICA_USER_SETTING_TAG = "FEDML_MODEL_ENDPOINT_REPLICA_USER_SETTING_TAG-"

    # For keeping track scale down decisions state.
    FEDML_MODEL_ENDPOINT_SCALING_DOWN_DECISION_TIME_TAG = "FEDML_MODEL_ENDPOINT_SCALING_DOWN_DECISION_TIME_TAG-"

    # On the worker
    FEDML_MODEL_REPLICA_GPU_IDS_TAG = "FEDML_MODEL_REPLICA_GPU_IDS_TAG-"

    FEDML_KEY_COUNT_PER_SCAN = 1000

    FEDML_PENDING_REQUESTS_COUNTER = "FEDML_PENDING_REQUESTS_COUNTER"

    def __init__(self):
        if not hasattr(self, "redis_pool"):
            self.redis_pool = None
        if not hasattr(self, "redis_connection"):
            self.redis_connection = None
        if not hasattr(self, "model_deployment_db"):
            self.model_deployment_db = FedMLModelDatabase().get_instance()
            self.model_deployment_db.create_table()
        self.redis_addr, self.redis_port, self.redis_password = None, None, None

    def setup_redis_connection(self, redis_addr, redis_port, redis_password="fedml_default"):
        _, env_redis_addr, env_redis_port, env_redis_pwd, disable_redis = \
            SchedulerConstants.get_redis_and_infer_host_env_addr()
        redis_addr = env_redis_addr if env_redis_addr is not None else redis_addr
        redis_addr = "localhost" if redis_addr is not None and redis_addr == "local" else redis_addr
        redis_port = env_redis_port if env_redis_port is not None else redis_port
        redis_password = env_redis_pwd if env_redis_pwd is not None else redis_password
        if disable_redis is not None:
            return False

        is_connected = False
        try:
            if redis_password is None or redis_password == "" or redis_password == "fedml_default":
                self.redis_pool = redis.ConnectionPool(host=redis_addr, port=int(redis_port), decode_responses=True)
            else:
                self.redis_pool = redis.ConnectionPool(host=redis_addr, port=int(redis_port),
                                                       password=redis_password, decode_responses=True)
            self.redis_connection = redis.Redis(
                connection_pool=self.redis_pool, socket_connect_timeout=SchedulerConstants.REDIS_CONN_TIMEOUT)
            self.redis_connection.set("FEDML_TEST_KEYS", "TEST")
            is_connected = True
        except Exception as e:
            is_connected = False

        if not is_connected:
            is_connected = self.setup_public_redis_connection()

        return is_connected

    def setup_public_redis_connection(self):
        is_connected = False
        try:
            self.redis_pool = redis.ConnectionPool(
                host=SchedulerConstants.get_public_redis_addr(), port=SchedulerConstants.PUBLIC_REDIS_PORT,
                password=SchedulerConstants.PUBLIC_REDIS_PASSWORD, decode_responses=True)
            self.redis_connection = redis.Redis(
                connection_pool=self.redis_pool, socket_connect_timeout=SchedulerConstants.REDIS_CONN_TIMEOUT)
            self.redis_connection.set("FEDML_TEST_KEYS", "TEST")
            is_connected = True
        except Exception as e:
            pass

        return is_connected

    def set_redis_params(self, redis_addr="local", redis_port=6379, redis_password="fedml_default"):
        self.redis_addr, self.redis_port, self.redis_password = redis_addr, redis_port, redis_password
        if self.redis_pool is None:
            if redis_addr is None or redis_addr == "local":
                self.setup_redis_connection("localhost", redis_port, redis_password)
            else:
                self.setup_redis_connection(redis_addr, redis_port, redis_password)

    def get_redis_params(self):
        if any([self.redis_addr is None, self.redis_port is None, self.redis_password is None]):
            raise RuntimeError("Redis parameters are not set.")
        else:
            return self.redis_addr, self.redis_port, self.redis_password

    @staticmethod
    def get_instance(redis_addr="local", redis_port=6379):
        return FedMLModelCache()

    def set_user_setting_replica_num(self, end_point_id,
                                     end_point_name: str, model_name: str, model_version: str,
                                     replica_num: int, enable_auto_scaling: bool = False,
                                     scale_min: int = 0, scale_max: int = 0, state: str = "UNKNOWN",
                                     target_queries_per_replica: int = 60, aggregation_window_size_seconds: int = 60,
                                     scale_down_delay_seconds: int = 120, timeout_s: int = 30
                                     ) -> bool:
        """
        Key: FEDML_MODEL_ENDPOINT_REPLICA_USER_SETTING_TAG--<end_point_id>
        Value: {
            "endpoint_name": end_point_name,
            "model_name": model_name,
            "model_version": model_version,
            "replica_num": replica_num,
            "enable_auto_scaling": enable_auto_scaling,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "state": state
        }
        replica_num is used for manual scale.
        scale_min and scale_max are used for auto-scaling.

        state should in "UNKNOWN", "DEPLOYED", "DEPLOYING".
        """
        assert state in ["UNKNOWN", "DEPLOYED", "DEPLOYING"]
        replica_num_dict = {
            "endpoint_id": end_point_id, "endpoint_name": end_point_name, "model_name": model_name,
            "model_version": model_version, "replica_num": replica_num, "enable_auto_scaling": enable_auto_scaling,
            "scale_min": scale_min, "scale_max": scale_max, "state": state,
            "target_queries_per_replica": target_queries_per_replica,
            "aggregation_window_size_seconds": aggregation_window_size_seconds,
            "scale_down_delay_seconds": scale_down_delay_seconds,
            ServerConstants.INFERENCE_REQUEST_TIMEOUT_KEY: timeout_s
        }
        try:
            self.redis_connection.set(self.get_user_setting_replica_num_key(end_point_id), json.dumps(replica_num_dict))
        except Exception as e:
            logging.error(e)
            return False
        return True

    def update_user_setting_replica_num(self, end_point_id: str, state: str = "UNKNOWN") -> bool:
        assert state in ["UNKNOWN", "DEPLOYED", "DEPLOYING"]
        # Get existed value
        try:
            replica_num_dict = self.redis_connection.get(self.get_user_setting_replica_num_key(end_point_id))
            replica_num_dict = json.loads(replica_num_dict)
        except Exception as e:
            logging.error(e)
            return False

        # Update the state
        replica_num_dict["state"] = state

        # Set the new value
        try:
            self.redis_connection.set(self.get_user_setting_replica_num_key(end_point_id), json.dumps(replica_num_dict))
        except Exception as e:
            logging.error(e)
            return False
        return True

    def get_all_endpoints_user_setting(self) -> List[dict]:
        """
        Return a list of dict, each dict is the user setting of an endpoint.
        """
        user_setting_list = list()
        try:
            key_pattern = "{}*".format(self.get_user_setting_replica_num_key(""))
            user_setting_keys = self.redis_connection.keys(pattern=key_pattern)
            for key in user_setting_keys:
                user_setting = self.redis_connection.get(key)
                user_setting_list.append(json.loads(user_setting))
        except Exception as e:
            logging.error(e)
        return user_setting_list

    @staticmethod
    def get_user_setting_replica_num_key(end_point_id):
        return "{}-{}".format(FedMLModelCache.FEDML_MODEL_ENDPOINT_REPLICA_USER_SETTING_TAG, end_point_id)

    def set_deployment_result(self, end_point_id, end_point_name,
                              model_name, model_version, device_id, deployment_result, replica_no):
        result_dict = {"cache_device_id": device_id, "cache_replica_no": replica_no, "result": deployment_result}
        try:
            # Delete old result using (e_id, end_point_name, model_name, device_id, replica_no)
            # In this list, find the result's complete record, delete it.
            result_list = self.redis_connection.lrange(
                self.get_deployment_result_key(end_point_id, end_point_name, model_name), 0, -1)
            for result_item in result_list:
                res_device_id, res_replica_no, res_payload = self.get_result_item_info(result_item)
                if res_device_id == device_id and res_replica_no == replica_no:
                    self.redis_connection.lrem(
                        self.get_deployment_result_key(end_point_id, end_point_name, model_name), 0, result_item)

            # Append the new result to the list
            self.redis_connection.rpush(
                self.get_deployment_result_key(end_point_id, end_point_name, model_name), json.dumps(result_dict))
        except Exception as e:
            pass
        self.model_deployment_db.set_deployment_result(end_point_id, end_point_name,
                                                       model_name, model_version,
                                                       device_id, deployment_result, replica_no)

    def set_deployment_status(self, end_point_id, end_point_name,
                              model_name, model_version, device_id, deployment_status, replica_no):
        status_dict = {"cache_device_id": device_id, "status": deployment_status}
        try:
            # rpush could tolerate the same e_id, d_id with different r_no
            self.redis_connection.rpush(self.get_deployment_status_key(end_point_id, end_point_name, model_name),
                                        json.dumps(status_dict))
        except Exception as e:
            pass
        self.model_deployment_db.set_deployment_status(end_point_id, end_point_name,
                                                       model_name, model_version,
                                                       device_id, deployment_status, replica_no)

    def delete_deployment_status(self, element: str, end_point_id, end_point_name, model_name):
        self.redis_connection.lrem(self.get_deployment_status_key(end_point_id, end_point_name, model_name),
                                   0, element)
        device_id, _ = self.get_status_item_info(element)
        self.model_deployment_db.delete_deployment_result(device_id, end_point_id, end_point_name, model_name)

    def delete_deployment_result(self, element: str, end_point_id, end_point_name, model_name):
        try:
            self.redis_connection.lrem(self.get_deployment_result_key(end_point_id, end_point_name, model_name),
                                       0, element)
        except Exception as e:
            pass

        device_id, replica_no, _ = self.get_result_item_info(element)
        self.model_deployment_db.delete_deployment_result_with_device_id_and_rank(
            end_point_id, end_point_name, model_name, device_id, replica_rank=replica_no-1)

        return

    def delete_deployment_result_with_device_id_and_replica_no(self, end_point_id, end_point_name, model_name,
                                                               device_id, replica_no_to_delete):
        result_item_found = None

        result_list = self.get_deployment_result_list(
            end_point_id, end_point_name, model_name)

        for result_item in result_list:
            cache_device_id, cache_replica_no, result_payload = (
                self.get_result_item_info(result_item))

            if str(cache_device_id) == str(device_id) and cache_replica_no == replica_no_to_delete:
                result_item_found = result_item
                break

        # Delete the replica element
        if result_item_found is not None:
            self.delete_deployment_result(
                result_item_found, end_point_id, end_point_name, model_name)

    def get_deployment_result_list(self, end_point_id, end_point_name, model_name):
        try:
            result_list = self.redis_connection.lrange(
                self.get_deployment_result_key(end_point_id, end_point_name, model_name), 0, -1)
        except Exception as e:
            logging.info(e)
            result_list = None

        if result_list is None or len(result_list) <= 0:
            result_list = self.model_deployment_db.get_deployment_result_list(end_point_id, end_point_name, model_name)
            try:
                for result in result_list:
                    self.redis_connection.rpush(self.get_deployment_result_key(end_point_id, end_point_name, model_name),
                                                json.dumps(result))
            except Exception as e:
                logging.info(e)
                pass
        return result_list

    def get_all_deployment_result_list(self):
        result_list = list()
        try:
            key_pattern = "{}*".format(self.FEDML_MODEL_DEPLOYMENT_RESULT_TAG)
            result_keys = self.redis_connection.keys(pattern=key_pattern)
            for key in result_keys:
                result_list.extend(self.redis_connection.lrange(key, 0, -1))
        except Exception as e:
            logging.error(e)
        # TODO(Raphael): Use Sqlite for the replica backup

        return result_list

    def get_deployment_result_list_size(self, end_point_id, end_point_name, model_name):
        result_list = self.get_deployment_result_list(end_point_id, end_point_name, model_name)
        return len(result_list)

    def get_deployment_status_list(self, end_point_id, end_point_name, model_name):
        try:
            status_list = self.redis_connection.lrange(self.get_deployment_status_key(end_point_id, end_point_name, model_name), 0, -1)
        except Exception as e:
            status_list = None

        if status_list is None or len(status_list) <= 0:
            status_list = self.model_deployment_db.get_deployment_status_list(end_point_id, end_point_name, model_name)
            try:
                for status in status_list:
                    self.redis_connection.rpush(self.get_deployment_status_key(end_point_id, end_point_name, model_name),
                                                json.dumps(status))
            except Exception as e:
                pass
        return status_list

    def get_deployment_status_list_size(self, end_point_id, end_point_name, model_name):
        status_list = self.get_deployment_status_list(end_point_id, end_point_name, model_name)
        return len(status_list)

    def get_status_item_info(self, status_item):
        status_item_json = json.loads(status_item)
        if isinstance(status_item_json, str):
            status_item_json = json.loads(status_item_json)
        device_id = status_item_json["cache_device_id"]
        if isinstance(status_item_json["status"], str):
            status_payload = json.loads(status_item_json["status"])
        else:
            status_payload = status_item_json["status"]
        return device_id, status_payload

    def get_result_item_info(self, result_item):
        result_item_json = json.loads(result_item)
        if isinstance(result_item_json, str):
            result_item_json = json.loads(result_item_json)

        device_id = result_item_json["cache_device_id"]
        replica_no = result_item_json.get("cache_replica_no", 1)    # Compatible with the old version

        if isinstance(result_item_json["result"], str):
            result_payload = json.loads(result_item_json["result"])
        else:
            result_payload = result_item_json["result"]
        return device_id, replica_no, result_payload

    def get_idle_device(self, end_point_id, end_point_name,
                        model_name, model_version,
                        check_end_point_status=True, limit_specific_model_version=False):
        # Deprecated the model status logic, query directly from the deployment result list
        idle_device_list = list()

        result_list = self.get_deployment_result_list(end_point_id, end_point_name, model_name)

        for result_item in result_list:
            device_id, _, result_payload = self.get_result_item_info(result_item)
            found_end_point_id = result_payload["end_point_id"]
            found_end_point_name = result_payload["end_point_name"]
            found_model_name = result_payload["model_name"]
            found_model_version = result_payload["model_version"]

            if (str(found_end_point_id) == str(end_point_id) and found_end_point_name == end_point_name and
                    found_model_name == model_name and
                    (not limit_specific_model_version or found_model_version == model_version)):
                if "model_status" in result_payload and result_payload["model_status"] == "DEPLOYED":
                    idle_device_list.append({"device_id": device_id, "end_point_id": end_point_id})

        logging.info(f"{len(idle_device_list)} devices this model has on it: {idle_device_list}")

        if len(idle_device_list) <= 0:
            return None, None

        # # Randomly shuffle
        # shuffle the list of deployed devices and get the first one as the target idle device.
        # if len(idle_device_list) <= 0:
        #     return None, None
        # shuffle(idle_device_list)
        # idle_device_dict = idle_device_list[0]

        # Round Robin
        total_device_num = len(idle_device_list)
        redis_round_robin_key = self.get_round_robin_prev_device(end_point_id, end_point_name, model_name, model_version)

        selected_device_index = 0
        try:
            if self.redis_connection.exists(redis_round_robin_key):
                selected_device_index = int(self.redis_connection.get(redis_round_robin_key))
                selected_device_index %= total_device_num
            else:
                selected_device_index = 0
            next_selected_device_index = (selected_device_index + 1) % total_device_num
            self.redis_connection.set(redis_round_robin_key, str(next_selected_device_index))
        except Exception as e:
            logging.info("Inference Device selection Failed:")
            logging.info(e)

        logging.info(f"Using Round Robin, the device index is {selected_device_index}")
        idle_device_dict = idle_device_list[selected_device_index]

        # Note that within the same endpoint_id, there could be one device with multiple same models
        same_model_device_rank = 0
        start = selected_device_index
        while(start != 0 and idle_device_list[start]["device_id"] == idle_device_list[start-1]["device_id"]):
            start -= 1
            same_model_device_rank += 1

        # Find deployment result from the target idle device.
        try:
            for result_item in result_list:
                logging.info("enter the for loop")
                device_id, _, result_payload = self.get_result_item_info(result_item)
                found_end_point_id = result_payload["end_point_id"]
                found_end_point_name = result_payload["end_point_name"]
                found_model_status = result_payload["model_status"]

                if found_model_status != "DEPLOYED":
                    continue

                if str(found_end_point_id) == str(idle_device_dict["end_point_id"]) \
                        and device_id == idle_device_dict["device_id"]:
                    if same_model_device_rank > 0:
                        same_model_device_rank -= 1
                        continue
                    logging.info(f"The chosen device is {device_id}")
                    return result_payload, device_id
        except Exception as e:
            logging.info(str(e))

        return None, None

    def get_latest_version(self, status_list):
        latest_version = None
        latest_version_int = -1
        for status_item in status_list:
            try:
                _, status_payload = self.get_status_item_info(status_item)
                model_version = status_payload["model_version"]
                prefix = model_version.split("-")[0]    # version-date
                prefix_int = int(prefix[1:])    # v12 -> 12

                if latest_version is None:
                    latest_version = model_version
                    latest_version_int = prefix_int
                elif prefix_int > latest_version_int:
                    latest_version = model_version
                    latest_version_int = prefix_int
            except Exception as e:
                pass

        return latest_version

    def get_deployment_result_with_device_id(self, end_point_id, end_point_name, model_name, device_id):
        """"
        TODO: Return multiple replicas' result for the same device_id
        """
        try:
            result_list = self.get_deployment_result_list(end_point_id, end_point_name, model_name)
            for result_item in result_list:
                result_device_id, _, result_payload = self.get_result_item_info(result_item)
                found_end_point_id = result_payload["end_point_id"]

                end_point_activated = self.get_end_point_activation(found_end_point_id)
                if str(found_end_point_id) == str(end_point_id) and str(result_device_id) == str(device_id):
                    return result_payload, end_point_activated
        except Exception as e:
            logging.info(e)

        return None, False

    def set_end_point_status(self, end_point_id, end_point_name, status):
        try:
            self.redis_connection.set(self.get_end_point_status_key(end_point_id), status)
        except Exception as e:
            pass
        self.model_deployment_db.set_end_point_status(end_point_id, end_point_name, status)

    def get_end_point_status(self, end_point_id):
        status = None
        try:
            if self.redis_connection.exists(self.get_end_point_status_key(end_point_id)):
                status = self.redis_connection.get(self.get_end_point_status_key(end_point_id))
        except Exception as e:
            status = None

        if status is None:
            status = self.model_deployment_db.get_end_point_status(end_point_id)
            if status is not None:
                try:
                    self.redis_connection.set(self.get_end_point_status_key(end_point_id), status)
                except Exception as e:
                    pass
            return status
        return status

    def set_end_point_activation(self, end_point_id, end_point_name, activate_status):
        status = 1 if activate_status else 0
        try:
            self.redis_connection.set(self.get_end_point_activation_key(end_point_id), status)
        except Exception as e:
            pass
        self.model_deployment_db.set_end_point_activation(end_point_id, end_point_name, status)

    def set_replica_gpu_ids(self, end_point_id, end_point_name, model_name, device_id, replica_no, gpu_ids):
        # Convert the list to string
        try:
            self.redis_connection.set(self.get_replica_gpu_ids_key(end_point_id, end_point_name,
                                                                   model_name, device_id, replica_no), str(gpu_ids))
        except Exception as e:
            print(e)
            logging.error(e)

        # TODO: Use Sqlite for the replica backup

    def delete_all_replica_gpu_ids(self, end_point_id, end_point_name, model_name, device_id):
        prefix_key = self.get_replica_gpu_ids_key(end_point_id, end_point_name, model_name, device_id, 1)[:-2]
        for key in self.redis_connection.scan_iter(prefix_key + "*"):
            self.redis_connection.delete(key)
        # TODO(Raphael): Delete the backup in Sqlite

    def get_replica_gpu_ids(self, end_point_id, end_point_name, model_name, device_id, replica_no):
        try:
            if self.redis_connection.exists(self.get_replica_gpu_ids_key(end_point_id, end_point_name,
                                                                         model_name, device_id, replica_no)):
                return self.redis_connection.get(self.get_replica_gpu_ids_key(end_point_id, end_point_name,
                                                                              model_name, device_id, replica_no))
        except Exception as e:
            pass
        return None

    def delete_end_point(self, end_point_id, end_point_name, model_name, model_version, device_id=None):
        # Device id is either deploy master or deploy worker
        try:
            logging.info("Will Delete the related redis keys permanently")
            self.redis_connection.expire(self.get_deployment_result_key(end_point_id, end_point_name, model_name), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_deployment_status_key(end_point_id, end_point_name, model_name), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_monitor_metrics_key(end_point_id, end_point_name, model_name, model_version), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_deployment_token_key(end_point_id, end_point_name, model_name), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)

            any_version_round_robin_key = self.get_round_robin_prev_device_any_version(end_point_id, end_point_name, model_name)
            for key in self.redis_connection.scan_iter(any_version_round_robin_key + "*"):
                self.redis_connection.expire(key, ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)

            self.redis_connection.expire(self.get_deployment_device_info_key(end_point_id), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_end_point_activation_key(end_point_id), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_end_point_status_key(end_point_id), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_user_setting_replica_num_key(end_point_id), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)

            # Delete all replicas gpu ids
            matched_prefix_replica = self.get_replica_gpu_ids_key_all_replicas(end_point_id, end_point_name, model_name, device_id)
            for key in self.redis_connection.scan_iter(matched_prefix_replica + "*"):
                self.redis_connection.expire(key, ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)

                logging.info(f"Those keys are deleted: {key}")

            # Delete the compute gpu cache
            self.redis_connection.expire(ComputeGpuCache.get_run_total_num_gpus_key(end_point_id), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(ComputeGpuCache.get_run_total_num_gpus_key(end_point_id), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(ComputeGpuCache.get_run_device_ids_key(end_point_id), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(ComputeGpuCache.get_edge_model_id_map_key(end_point_id), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)

            logging.info(f"Those keys are deleted:"
                         f"{ComputeGpuCache.get_endpoint_run_id_map_key(end_point_id)}, "
                         f"{ComputeGpuCache.get_run_total_num_gpus_key(end_point_id)}, "
                         f"{ComputeGpuCache.get_run_total_num_gpus_key(end_point_id)}, "
                         f"{ComputeGpuCache.get_run_device_ids_key(end_point_id)}, "
                         f"{ComputeGpuCache.get_edge_model_id_map_key(end_point_id)}")

        except Exception as e:
            logging.error(f"error when deleting the redis keys: {e}")
            pass

    def get_end_point_activation(self, end_point_id):
        activation = False
        try:
            if self.redis_connection.exists(self.get_end_point_activation_key(end_point_id)):
                activation = self.redis_connection.get(self.get_end_point_activation_key(end_point_id))
        except Exception as e:
            activation = False
        return activation

    def get_end_point_full_key_by_id(self, end_point_id):
        # e.g. FEDML_MODEL_DEPLOYMENT_RESULT--1234-dummy_endpoint_name-dummy_model_name
        target_prefix = f"{FedMLModelCache.FEDML_MODEL_DEPLOYMENT_RESULT_TAG}-{end_point_id}-*"
        status_list = list()
        for key in self.redis_connection.scan_iter(target_prefix):
            status_list.append(key)
        if len(status_list) <= 0:
            return None
        status_key = status_list[0]
        return status_key

    def set_end_point_device_info(self, end_point_id, end_point_name, device_info):
        '''
        Currently all the device info is stored in one key, which is a string.
        This string can be parsed into a list of device info.
        '''
        try:
            self.redis_connection.set(self.get_deployment_device_info_key(end_point_id), device_info)
        except Exception as e:
            pass
        self.model_deployment_db.set_end_point_device_info(end_point_id, end_point_name, device_info)

    def get_end_point_device_info(self, end_point_id):
        device_info = None
        try:
            if self.redis_connection.exists(self.get_deployment_device_info_key(end_point_id)):
                device_info = self.redis_connection.get(self.get_deployment_device_info_key(end_point_id))
        except Exception as e:
            device_info = None

        if device_info is None:
            device_info = self.model_deployment_db.get_end_point_device_info(end_point_id)
            if device_info is not None:
                try:
                    self.redis_connection.set(self.get_deployment_device_info_key(end_point_id), device_info)
                except Exception as e:
                    pass

        return device_info

    def delete_end_point_device_info(self, end_point_id, end_point_name, edge_id_list_to_delete):
        '''
        Since the device info is stored in one key, we need to first delete the device info from the existing one.
        '''
        device_objs = FedMLModelCache.get_instance().get_end_point_device_info(end_point_id)

        if device_objs is None:
            raise Exception("The device list in local redis is None")
        else:
            total_device_objs_list = json.loads(device_objs)
            for device_obj in total_device_objs_list:
                if device_obj["id"] in edge_id_list_to_delete:
                    total_device_objs_list.remove(device_obj)

        # Dumps the new record (after deletion) to the redis
        FedMLModelCache.get_instance().set_end_point_device_info(
            end_point_id, end_point_name, json.dumps(total_device_objs_list))

    def add_end_point_device_info(self, end_point_id, end_point_name, new_device_info):
        '''
        Since the device info is stored in one key, we need to append the new device info to the existing one.
        '''
        device_objs = FedMLModelCache.get_instance().get_end_point_device_info(end_point_id)

        if device_objs is None:
            raise Exception("The device list in local redis is None")
        else:
            total_device_objs_list = json.loads(device_objs)
            new_device_info_json = json.loads(new_device_info)
            total_device_objs_list.append(new_device_info_json)

        FedMLModelCache.get_instance().set_end_point_device_info(
            end_point_id, end_point_name, json.dumps(total_device_objs_list))

    def set_end_point_token(self, end_point_id, end_point_name, model_name, token):
        try:
            if self.redis_connection.exists(self.get_deployment_token_key(end_point_id, end_point_name, model_name)):
                return
            self.redis_connection.set(self.get_deployment_token_key(end_point_id, end_point_name, model_name), token)
        except Exception as e:
            pass

        self.model_deployment_db.set_end_point_token(end_point_id, end_point_name, model_name, token)

    def get_end_point_token(self, end_point_id, end_point_name, model_name):
        token = None
        try:
            if self.redis_connection.exists(self.get_deployment_token_key(end_point_id, end_point_name, model_name)):
                token = self.redis_connection.get(self.get_deployment_token_key(end_point_id, end_point_name, model_name))
        except Exception as e:
            token = None

        if token is None:
            token = self.model_deployment_db.get_end_point_token(end_point_id, end_point_name, model_name)
            if token is not None:
                try:
                    self.redis_connection.set(self.get_deployment_token_key(end_point_id, end_point_name, model_name), token)
                except Exception as e:
                    pass

        return token

    def get_end_point_token_with_eid(self, end_point_id):
        # Only support redis for now
        token = None
        try:
            prefix = self.get_deployment_token_key_eid(end_point_id)
            for key in self.redis_connection.scan_iter(prefix + "*"):
                token = self.redis_connection.get(key)
                break
        except Exception as e:
            token = None

        return token

    def get_endpoint_devices_replica_num(self, end_point_id):
        """
        Return a endpoint_devices_replica_num dict {id1: 1, id2: 1}, if not exist, return None
        """
        try:
            replica_num = self.redis_connection.get(
                self.get_endpoint_replica_num_key(end_point_id))
        except Exception as e:
            replica_num = None
        # TODO: Use Sqlite for the replica backup

        return replica_num

    def get_deployment_result_key(self, end_point_id, end_point_name, model_name):
        return "{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_DEPLOYMENT_RESULT_TAG, end_point_id, end_point_name, model_name)

    def get_deployment_status_key(self, end_point_id, end_point_name, model_name):
        return "{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_DEPLOYMENT_STATUS_TAG, end_point_id, end_point_name, model_name)

    def get_end_point_status_key(self, end_point_id):
        return "{}{}".format(FedMLModelCache.FEDML_MODEL_END_POINT_STATUS_TAG, end_point_id)

    @staticmethod
    def get_end_point_activation_key(end_point_id):
        return "{}{}".format(FedMLModelCache.FEDML_MODEL_END_POINT_ACTIVATION_TAG, end_point_id)

    def get_deployment_device_info_key(self, end_point_id):
        return "{}{}".format(FedMLModelCache.FEDML_MODEL_DEVICE_INFO_TAG, end_point_id)

    @staticmethod
    def get_deployment_token_key(end_point_id, end_point_name, model_name):
        return "{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_END_POINT_TOKEN_TAG, end_point_id, end_point_name, model_name)

    @staticmethod
    def get_deployment_token_key_eid(end_point_id):
        return "{}-{}".format(FedMLModelCache.FEDML_MODEL_END_POINT_TOKEN_TAG, end_point_id)

    def get_round_robin_prev_device(self, end_point_id, end_point_name, model_name, version):
        return "{}-{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_ROUND_ROBIN_PREVIOUS_DEVICE_TAG, end_point_id, end_point_name, model_name, version)

    @staticmethod
    def get_round_robin_prev_device_any_version(endpoint_id, endpoint_name, model_name):
        return "{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_ROUND_ROBIN_PREVIOUS_DEVICE_TAG, endpoint_id,
                                    endpoint_name, model_name)

    def get_endpoint_replica_num_key(self, end_point_id):
        return "{}-{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_ENDPOINT_REPLICA_NUM_TAG, end_point_id, "replica_num", "key")

    @staticmethod
    def get_replica_gpu_ids_key(end_point_id, end_point_name, model_name, device_id, replica_no):
        return "{}-{}-{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_REPLICA_GPU_IDS_TAG, end_point_id,
                                          end_point_name, model_name, device_id, replica_no)

    @staticmethod
    def get_replica_gpu_ids_key_all_replicas(end_point_id, end_point_name, model_name, device_id):
        return "{}-{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_REPLICA_GPU_IDS_TAG, end_point_id,
                                       end_point_name, model_name, device_id)

    def set_monitor_metrics(self, end_point_id, end_point_name,
                            model_name, model_version,
                            total_latency, avg_latency, current_latency,
                            total_request_num, current_qps,
                            avg_qps, timestamp, device_id):
        metrics_dict = {"total_latency": total_latency, "avg_latency": avg_latency, "current_latency": current_latency,
                        "total_request_num": total_request_num, "current_qps": current_qps,
                        "avg_qps": avg_qps, "timestamp": timestamp, "device_id": device_id}
        try:
            self.redis_connection.rpush(self.get_monitor_metrics_key(end_point_id, end_point_name, model_name, model_version),
                                        json.dumps(metrics_dict))
        except Exception as e:
            pass
        self.model_deployment_db.set_monitor_metrics(end_point_id, end_point_name,
                                                     model_name, model_version,
                                                     total_latency, avg_latency, current_latency,
                                                     total_request_num, current_qps,
                                                     avg_qps, timestamp, device_id)

    def get_latest_monitor_metrics(self, end_point_id, end_point_name, model_name, model_version):
        try:
            if self.redis_connection.exists(self.get_monitor_metrics_key(end_point_id, end_point_name, model_name, model_version)):
                return self.redis_connection.lindex(self.get_monitor_metrics_key(end_point_id, end_point_name, model_name, model_version), -1)
        except Exception as e:
            pass

        metrics_dict = self.model_deployment_db.get_latest_monitor_metrics(end_point_id, end_point_name, model_name, model_version)
        if metrics_dict is not None:
            try:
                self.redis_connection.rpush(self.get_monitor_metrics_key(end_point_id, end_point_name, model_name, model_version),
                                            metrics_dict)
            except Exception as e:
                pass

        return metrics_dict

    def get_monitor_metrics_item(self, end_point_id, end_point_name, model_name, model_version, index):
        try:
            if self.redis_connection.exists(self.get_monitor_metrics_key(end_point_id, end_point_name, model_name, model_version)):
                metrics_item = self.redis_connection.lindex(self.get_monitor_metrics_key(end_point_id, end_point_name, model_name,
                                                                                     model_version), index)
                return metrics_item, index+1
        except Exception as e:
            pass

        metrics_dict = self.model_deployment_db.get_monitor_metrics_item(end_point_id, end_point_name, model_name, model_version, index)
        if metrics_dict is not None:
            try:
                self.redis_connection.rpush(self.get_monitor_metrics_key(end_point_id, end_point_name, model_name, model_version),
                                            metrics_dict)
            except Exception as e:
                pass
            return metrics_dict, index+1

        return None, 0

    def get_metrics_item_info(self, metrics_item):
        metrics_item_json = json.loads(metrics_item)
        total_latency = metrics_item_json["total_latency"]
        avg_latency = metrics_item_json["avg_latency"]
        total_request_num = metrics_item_json["total_request_num"]
        current_qps = metrics_item_json["current_qps"]

        # For the old version, the current_latency is not available
        current_latency = metrics_item_json.get("current_latency", avg_latency)

        avg_qps = metrics_item_json["avg_qps"]
        timestamp = metrics_item_json["timestamp"]
        device_id = metrics_item_json["device_id"]
        return total_latency, avg_latency, current_latency, total_request_num, current_qps, avg_qps, timestamp, device_id

    def get_monitor_metrics_key(self, end_point_id, end_point_name, model_name, model_version):
        return "{}{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_DEPLOYMENT_MONITOR_TAG,
                                      end_point_id, end_point_name, model_name, model_version)

    def get_endpoint_metrics(self,
                             end_point_id,
                             k_recent=None) -> List[Any]:
        model_deployment_monitor_metrics = list()
        try:
            key_pattern = "{}*{}*".format(
                self.FEDML_MODEL_DEPLOYMENT_MONITOR_TAG,
                end_point_id)
            model_deployment_monitor_endpoint_key = \
                self.redis_connection.keys(pattern=key_pattern)
            # Since the reply is a list, we need to make sure the list
            # is non-empty otherwise the index will raise an error.
            if model_deployment_monitor_endpoint_key:
                model_deployment_monitor_endpoint_key = \
                    model_deployment_monitor_endpoint_key[0]

                # Set start and end index depending on the size of the
                # list and the requested number of most recent records.
                num_records = self.redis_connection.llen(
                    name=model_deployment_monitor_endpoint_key)
                # if k_most_recent is None, then fetch all by default.
                start, end = 0, -1
                # if k_most_recent is positive then fetch [-k_most_recent:]
                if k_recent and k_recent > 0:
                    start = num_records - k_recent
                model_deployment_monitor_metrics = \
                    self.redis_connection.lrange(
                        name=model_deployment_monitor_endpoint_key,
                        start=start,
                        end=end)
                model_deployment_monitor_metrics = [
                    json.loads(m) for m in model_deployment_monitor_metrics]

        except Exception as e:
            logging.error(e)

        return model_deployment_monitor_metrics

    def get_endpoint_replicas_results(self, endpoint_id) -> List[Any]:
        replicas_results = []
        try:
            key_pattern = "{}*{}*".format(
                self.FEDML_MODEL_DEPLOYMENT_RESULT_TAG,
                endpoint_id)
            model_deployment_result_keys = \
                self.redis_connection.keys(pattern=key_pattern)
            if model_deployment_result_keys:
                model_deployment_result_key = \
                    model_deployment_result_keys[0]
                replicas_results = \
                    self.redis_connection.lrange(
                        name=model_deployment_result_key,
                        start=0,
                        end=-1)
                # Format the result value to a properly formatted json.
                for replica_idx, replica in enumerate(replicas_results):
                    replicas_results[replica_idx] = json.loads(replica)
                    replicas_results[replica_idx]["result"] = \
                        json.loads(replicas_results[replica_idx]["result"])
            else:
                raise Exception("Function `get_endpoint_replicas_results` Key {} does not exist."
                                .format(key_pattern))

        except Exception as e:
            logging.error(e)

        return replicas_results

    def get_endpoint_settings(self, endpoint_id) -> Dict:
        endpoint_settings = {}
        try:
            key_pattern = "{}*{}*".format(
                self.FEDML_MODEL_ENDPOINT_REPLICA_USER_SETTING_TAG,
                endpoint_id)

            endpoint_settings_keys = \
                self.redis_connection.keys(pattern=key_pattern)

            if len(endpoint_settings_keys) > 0:
                endpoint_settings = \
                    self.redis_connection.get(endpoint_settings_keys[0])

                if not isinstance(endpoint_settings, dict):
                    endpoint_settings = json.loads(endpoint_settings)
            else:
                raise Exception("Function `get_endpoint_settings` Key {} does not exist."
                                .format(key_pattern))
        except Exception as e:
            logging.error(e)

        return endpoint_settings

    def delete_endpoint_metrics(self, endpoint_ids: list) -> bool:
        status = True
        for endpoint_id in endpoint_ids:
            try:
                key_pattern = "{}*{}*".format(
                    self.FEDML_MODEL_DEPLOYMENT_MONITOR_TAG,
                    endpoint_id)
                model_deployment_monitor_endpoint_keys = \
                    self.redis_connection.keys(pattern=key_pattern)
                for k in model_deployment_monitor_endpoint_keys:
                    self.redis_connection.delete(k)
            except Exception as e:
                logging.error(e)
                # False if an error occurred.
                status = False
        return status

    def set_endpoint_scaling_down_decision_time(self, end_point_id, timestamp) -> bool:
        status = True
        try:
            self.redis_connection.hset(
                self.FEDML_MODEL_ENDPOINT_SCALING_DOWN_DECISION_TIME_TAG,
                mapping={end_point_id: timestamp})
        except Exception as e:
            logging.error(e)
            status = False
        return status

    def get_endpoint_scaling_down_decision_time(self, end_point_id) -> int:
        try:
            scaling_down_decision_time = \
                self.redis_connection.hget(
                    self.FEDML_MODEL_ENDPOINT_SCALING_DOWN_DECISION_TIME_TAG,
                    end_point_id)
            if len(scaling_down_decision_time) > 0:
                scaling_down_decision_time = int(scaling_down_decision_time)
            else:
                scaling_down_decision_time = 0
        except Exception as e:
            scaling_down_decision_time = 0
            logging.error(e)

        return scaling_down_decision_time

    def exists_endpoint_scaling_down_decision_time(self, end_point_id) -> bool:
        # The hash exists returns an integer 0 (not found), 1 (found), hence we need
        # to cast it to a boolean value.
        return bool(self.redis_connection.hexists(
            self.FEDML_MODEL_ENDPOINT_SCALING_DOWN_DECISION_TIME_TAG,
            end_point_id))

    def delete_endpoint_scaling_down_decision_time(self, end_point_id) -> bool:
        return bool(self.redis_connection.hdel(
            self.FEDML_MODEL_ENDPOINT_SCALING_DOWN_DECISION_TIME_TAG,
            end_point_id))

    def get_pending_requests_counter(self, end_point_id) -> int:
        # If the endpoint does not exist inside the Hash collection, set its counter to 0.
        if self.redis_connection.hexists(self.FEDML_PENDING_REQUESTS_COUNTER, end_point_id):
            return int(self.redis_connection.hget(self.FEDML_PENDING_REQUESTS_COUNTER, end_point_id))
        return 0

    def update_pending_requests_counter(self, end_point_id, increase=False, decrease=False) -> int:
        if not self.redis_connection.hexists(self.FEDML_PENDING_REQUESTS_COUNTER, end_point_id):
            self.redis_connection.hset(self.FEDML_PENDING_REQUESTS_COUNTER, mapping={end_point_id: 0})
        if increase:
            self.redis_connection.hincrby(self.FEDML_PENDING_REQUESTS_COUNTER, end_point_id, 1)
        if decrease:
            # Careful on the negative, there is no native function for hash decreases.
            self.redis_connection.hincrby(self.FEDML_PENDING_REQUESTS_COUNTER, end_point_id, -1)
            # Making sure the counter never becomes negative!
            if self.get_pending_requests_counter(end_point_id) < 0:
                self.redis_connection.hset(self.FEDML_PENDING_REQUESTS_COUNTER, mapping={end_point_id: 0})
        return self.get_pending_requests_counter(end_point_id)
