import json
import logging

import redis

from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
from .device_model_db import FedMLModelDatabase
from fedml.core.common.singleton import Singleton


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

    # On the worker
    FEDML_MODEL_REPLICA_GPU_IDS_TAG = "FEDML_MODEL_REPLICA_GPU_IDS_TAG-"

    FEDML_KEY_COUNT_PER_SCAN = 1000

    def __init__(self):
        if not hasattr(self, "redis_pool"):
            self.redis_pool = None
        if not hasattr(self, "redis_connection"):
            self.redis_connection = None
        if not hasattr(self, "model_deployment_db"):
            self.model_deployment_db = FedMLModelDatabase().get_instance()
            self.model_deployment_db.create_table()

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
        if self.redis_pool is None:
            if redis_addr is None or redis_addr == "local":
                self.setup_redis_connection("localhost", redis_port, redis_password)
            else:
                self.setup_redis_connection(redis_addr, redis_port, redis_password)

    @staticmethod
    def get_instance(redis_addr="local", redis_port=6379):
        return FedMLModelCache()

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

    def delete_deployment_result(self, element: str, end_point_id, end_point_name, model_name):
        self.redis_connection.lrem(self.get_deployment_result_key(end_point_id, end_point_name, model_name),
                                   0, element)
        device_id, _, _ = self.get_result_item_info(element)
        self.model_deployment_db.delete_deployment_result(device_id, end_point_id, end_point_name, model_name)

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

        logging.info(f"{len(idle_device_list)} devices has this model on it: {idle_device_list}")

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

    def get_replica_gpu_ids(self, end_point_id, end_point_name, model_name, device_id, replica_no):
        try:
            if self.redis_connection.exists(self.get_replica_gpu_ids_key(end_point_id, end_point_name,
                                                                         model_name, device_id, replica_no)):
                return self.redis_connection.get(self.get_replica_gpu_ids_key(end_point_id, end_point_name,
                                                                              model_name, device_id, replica_no))
        except Exception as e:
            pass

    def delete_end_point(self, end_point_id, end_point_name, model_name, model_version):
        try:
            logging.info("Will Delete the related redis keys permanently")
            self.redis_connection.expire(self.get_deployment_result_key(end_point_id, end_point_name, model_name), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_deployment_status_key(end_point_id, end_point_name, model_name), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_monitor_metrics_key(end_point_id, end_point_name, model_name, model_version), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_deployment_token_key(end_point_id, end_point_name, model_name), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_round_robin_prev_device(end_point_id, end_point_name, model_name, model_version), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_deployment_device_info_key(end_point_id), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_end_point_activation_key(end_point_id), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
            self.redis_connection.expire(self.get_end_point_status_key(end_point_id), ServerConstants.MODEL_CACHE_KEY_EXPIRE_TIME)
        except Exception as e:
            logging.error(f"error when deleting the redis keys: {e}")
            pass

    def get_end_point_activation(self, end_point_id):
        # [Deprecated] activation logic is removed
        return True

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

    def get_end_point_activation_key(self, end_point_id):
        return "{}{}".format(FedMLModelCache.FEDML_MODEL_END_POINT_ACTIVATION_TAG, end_point_id)

    def get_deployment_device_info_key(self, end_point_id):
        return "{}{}".format(FedMLModelCache.FEDML_MODEL_DEVICE_INFO_TAG, end_point_id)

    def get_deployment_token_key(self, end_point_id, end_point_name, model_name):
        return "{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_END_POINT_TOKEN_TAG, end_point_id, end_point_name, model_name)

    def get_round_robin_prev_device(self, end_point_id, end_point_name, model_name, version):
        return "{}-{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_ROUND_ROBIN_PREVIOUS_DEVICE_TAG, end_point_id, end_point_name, model_name, version)

    def get_endpoint_replica_num_key(self, end_point_id):
        return "{}-{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_ENDPOINT_REPLICA_NUM_TAG, end_point_id, "replica_num", "key")

    @staticmethod
    def get_replica_gpu_ids_key(end_point_id, end_point_name, model_name, device_id, replica_no):
        return "{}-{}-{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_REPLICA_GPU_IDS_TAG, end_point_id,
                                          end_point_name, model_name, device_id, replica_no)

    def set_monitor_metrics(self, end_point_id, end_point_name,
                            model_name, model_version,
                            total_latency, avg_latency,
                            total_request_num, current_qps,
                            avg_qps, timestamp, device_id):
        metrics_dict = {"total_latency": total_latency, "avg_latency": avg_latency,
                        "total_request_num": total_request_num, "current_qps": current_qps,
                        "avg_qps": avg_qps, "timestamp": timestamp, "device_id": device_id}
        try:
            self.redis_connection.rpush(self.get_monitor_metrics_key(end_point_id, end_point_name, model_name, model_version),
                                        json.dumps(metrics_dict))
        except Exception as e:
            pass
        self.model_deployment_db.set_monitor_metrics(end_point_id, end_point_name,
                                                     model_name, model_version,
                                                     total_latency, avg_latency,
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
        avg_qps = metrics_item_json["avg_qps"]
        timestamp = metrics_item_json["timestamp"]
        device_id = metrics_item_json["device_id"]
        return total_latency, avg_latency, total_request_num, current_qps, avg_qps, timestamp, device_id

    def get_monitor_metrics_key(self,end_point_id, end_point_name, model_name, model_version):
        return "{}{}-{}-{}-{}".format(FedMLModelCache.FEDML_MODEL_DEPLOYMENT_MONITOR_TAG,
                                      end_point_id, end_point_name, model_name, model_version)


if __name__ == "__main__":
    _end_point_id_ = "4f63aa70-312e-4a9c-872d-cc6e8d95f95b"
    _end_point_name_ = "my-llm"
    _model_name_ = "my-model"
    _model_version_ = "v1"
    _status_list_ = FedMLModelCache.get_instance().get_deployment_status_list(_end_point_id_, _end_point_name_, _model_name_)
    _result_list_ = FedMLModelCache.get_instance().get_deployment_result_list(_end_point_id_, _end_point_name_, _model_name_)
    idle_result_payload = FedMLModelCache.get_instance().get_idle_device(_end_point_id_, _end_point_name_, _model_name_, _model_version_)
