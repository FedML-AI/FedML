import traceback
import logging
from .compute_utils import ComputeUtils
from .compute_gpu_db import ComputeGpuDatabase
from ..slave import client_constants


class ComputeGpuCache(object):

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
    FEDML_RUN_INFO_LOCK_TAG = "FEDML_RUN_INFO_LOCK_TAG-"
    FEDML_RUN_INFO_SYNC_LOCK_TAG = "FEDML_RUN_INFO_SYNC_LOCK_TAG-"
    FEDML_EDGE_ID_MODEL_DEVICE_ID_MAP_TAG = "FEDML_EDGE_ID_MODEL_DEVICE_ID_MAP_TAG-"
    FEDML_GLOBAL_ENDPOINT_RUN_ID_MAP_TAG = "FEDML_GLOBAL_ENDPOINT_RUN_ID_MAP_TAG-"

    def __init__(self, redis_connection):
        self.redis_connection = redis_connection
        ComputeGpuDatabase.get_instance().set_database_base_dir(client_constants.ClientConstants.get_database_dir())
        ComputeGpuDatabase.get_instance().create_table()

    def get_device_run_num_gpus(self, device_id, run_id):
        device_run_num_gpus = None
        try:
            if self.redis_connection.exists(self.get_device_run_num_gpus_key(device_id, run_id)):
                device_run_num_gpus = self.redis_connection.get(self.get_device_run_num_gpus_key(device_id, run_id))
        except Exception as e:
            pass

        if device_run_num_gpus is None:
            device_run_num_gpus = ComputeGpuDatabase.get_instance().get_device_run_num_gpus(device_id, run_id)
            try:
                self.redis_connection.set(self.get_device_run_num_gpus_key(device_id, run_id), device_run_num_gpus)
            except Exception as e:
                pass

        return device_run_num_gpus

    def delete_device_run_num_gpus(self, device_id, run_id):
        try:
            self.redis_connection.delete(self.get_device_run_num_gpus_key(device_id, run_id))
        except Exception as e:
            logging.error(f"Error deleting device_run_num_gpus: {e}, Traceback: {traceback.format_exc()}")

        ComputeGpuDatabase.get_instance().delete_device_run_num_gpus(device_id, run_id)

    def get_device_run_gpu_ids(self, device_id, run_id):
        device_run_gpu_ids = None
        try:
            if self.redis_connection.exists(self.get_device_run_gpu_ids_key(device_id, run_id)):
                device_run_gpu_ids = self.redis_connection.get(self.get_device_run_gpu_ids_key(device_id, run_id))
                if str(device_run_gpu_ids).strip() == "":
                    return None
        except Exception as e:
            pass

        if device_run_gpu_ids is None:
            device_run_gpu_ids = ComputeGpuDatabase.get_instance().get_device_run_gpu_ids(device_id, run_id)
            try:
                self.redis_connection.set(self.get_device_run_gpu_ids_key(device_id, run_id), device_run_gpu_ids)
            except Exception as e:
                pass

        if not device_run_gpu_ids:
            return None

        device_run_gpu_ids = self.map_str_list_to_int_list(device_run_gpu_ids.split(','))
        return device_run_gpu_ids

    def delete_device_run_gpu_ids(self, device_id, run_id):
        try:
            self.redis_connection.delete(self.get_device_run_gpu_ids_key(device_id, run_id))
        except Exception as e:
            logging.error(f"Error deleting device_run_gpu_ids: {e}, Traceback: {traceback.format_exc()}")

        ComputeGpuDatabase.get_instance().delete_device_run_gpu_ids(device_id, run_id)

    def get_device_available_gpu_ids(self, device_id):
        device_available_gpu_ids = None
        try:
            if self.redis_connection.exists(self.get_device_available_gpu_ids_key(device_id)):
                device_available_gpu_ids = self.redis_connection.get(self.get_device_available_gpu_ids_key(device_id))
                if str(device_available_gpu_ids).strip() == "":
                    return []
        except Exception as e:
            pass

        if device_available_gpu_ids is None:
            device_available_gpu_ids = ComputeGpuDatabase.get_instance().get_device_available_gpu_ids(device_id)
            try:
                self.redis_connection.set(self.get_device_available_gpu_ids_key(device_id), device_available_gpu_ids)
            except Exception as e:
                pass

        if device_available_gpu_ids is not None and str(device_available_gpu_ids).strip() != "":
            device_available_gpu_ids = device_available_gpu_ids.split(',')
            device_available_gpu_ids = self.map_str_list_to_int_list(device_available_gpu_ids)
        else:
            return []

        return device_available_gpu_ids

    def get_device_total_num_gpus(self, device_id):
        device_total_num_gpus = None
        try:
            if self.redis_connection.exists(self.get_device_total_num_gpus_key(device_id)):
                device_total_num_gpus = self.redis_connection.get(self.get_device_total_num_gpus_key(device_id))
        except Exception as e:
            logging.error(f"Error getting device_total_num_gpus: {e}, Traceback: {traceback.format_exc()}")
            pass

        if device_total_num_gpus is None:
            device_total_num_gpus = ComputeGpuDatabase.get_instance().get_device_total_num_gpus(device_id)
            try:
                self.set_device_total_num_gpus(device_id, device_total_num_gpus)
            except Exception as e:
                logging.error(f"Error setting device_total_num_gpus: {e}, Traceback: {traceback.format_exc()}")
                pass

        return device_total_num_gpus

    def get_run_total_num_gpus(self, run_id):
        run_total_num_gpus = None
        try:
            if self.redis_connection.exists(self.get_run_total_num_gpus_key(run_id)):
                run_total_num_gpus = self.redis_connection.get(self.get_run_total_num_gpus_key(run_id))
        except Exception as e:
            pass

        if run_total_num_gpus is None:
            run_total_num_gpus = ComputeGpuDatabase.get_instance().get_run_total_num_gpus(run_id)
            try:
                self.redis_connection.set(self.get_run_total_num_gpus_key(run_id), run_total_num_gpus)
            except Exception as e:
                pass

        return run_total_num_gpus

    def get_run_device_ids(self, run_id):
        run_device_ids = None
        try:
            if self.redis_connection.exists(self.get_run_device_ids_key(run_id)):
                run_device_ids = self.redis_connection.get(self.get_run_device_ids_key(run_id))
                if str(run_device_ids).strip() == "":
                    return None

        except Exception as e:
            pass

        if run_device_ids is None:
            run_device_ids = ComputeGpuDatabase.get_instance().get_run_device_ids(run_id)
            try:
                self.redis_connection.set(self.get_run_device_ids_key(run_id), run_device_ids)
            except Exception as e:
                pass

        if run_device_ids is not None:
            run_device_ids = run_device_ids.split(',')

        return run_device_ids

    def get_edge_model_id_map(self, run_id):
        edge_id = None
        try:
            if self.redis_connection.exists(self.get_edge_model_id_map_key(run_id)):
                ids_map = self.redis_connection.get(self.get_edge_model_id_map_key(run_id))
                if str(ids_map).strip() == "":
                    return None, None, None
                ids_split = ids_map.split(',')
                if len(ids_split) != 3:
                    return None, None, None
                edge_id, model_master_device_id, model_slave_device_id = ids_split[0], ids_split[1], ids_split[2]
                return edge_id, model_master_device_id, model_slave_device_id
        except Exception as e:
            pass

        if edge_id is None:
            edge_id, model_master_device_id, model_slave_device_id = ComputeGpuDatabase.get_instance().get_edge_model_id_map(run_id)
            try:
                self.redis_connection.set(self.get_edge_model_id_map_key(run_id), f"{edge_id},{model_master_device_id},{model_slave_device_id}")
            except Exception as e:
                pass

        return edge_id, model_master_device_id, model_slave_device_id

    def get_endpoint_run_id_map(self, endpoint_id):
        # Map the endpoint_id (Deploy) to the run_id (Launch)
        # TODO(Raphael): Check if we can depreciate this function
        run_id = None
        try:
            if self.redis_connection.exists(self.get_endpoint_run_id_map_key(endpoint_id)):
                run_id = self.redis_connection.get(self.get_endpoint_run_id_map_key(endpoint_id))
                return run_id
        except Exception as e:
            pass

        if run_id is None:
            run_id = ComputeGpuDatabase.get_instance().get_endpoint_run_id_map(endpoint_id)
            try:
                self.redis_connection.set(self.get_endpoint_run_id_map_key(endpoint_id), f"{run_id}")
            except Exception as e:
                pass

        return run_id

    def set_device_run_num_gpus(self, device_id, run_id, num_gpus):
        try:
            self.redis_connection.set(self.get_device_run_num_gpus_key(device_id, run_id), num_gpus)
        except Exception as e:
            logging.error(f"Error setting device_run_num_gpus: {e}, Traceback: {traceback.format_exc()}")
            pass
        ComputeGpuDatabase.get_instance().set_device_run_num_gpus(device_id, run_id, num_gpus)

    def set_device_run_gpu_ids(self, device_id, run_id, gpu_ids):
        try:
            if gpu_ids is None:
                if self.redis_connection.exists(self.get_device_run_gpu_ids_key(device_id, run_id)):
                    self.redis_connection.delete(self.get_device_run_gpu_ids_key(device_id, run_id))
                return

            str_gpu_ids = self.map_list_to_str(gpu_ids)
            self.redis_connection.set(self.get_device_run_gpu_ids_key(device_id, run_id), str_gpu_ids)
        except Exception as e:
            logging.error(f"Error setting device_run_gpu_ids: {e}, Traceback: {traceback.format_exc()}")

        ComputeGpuDatabase.get_instance().set_device_run_gpu_ids(device_id, run_id, gpu_ids)

    def set_device_available_gpu_ids(self, device_id, gpu_ids):
        try:
            str_gpu_ids = self.map_list_to_str(gpu_ids)
            self.redis_connection.set(self.get_device_available_gpu_ids_key(device_id), str_gpu_ids)
        except Exception as e:
            pass

        ComputeGpuDatabase.get_instance().set_device_available_gpu_ids(device_id, gpu_ids)

    def set_device_total_num_gpus(self, device_id, num_gpus):
        try:
            self.redis_connection.set(self.get_device_total_num_gpus_key(device_id), num_gpus)
        except Exception as e:
            logging.error(f"Error setting device_total_num_gpus: {e}, Traceback: {traceback.format_exc()}")
            pass

        ComputeGpuDatabase.get_instance().set_device_total_num_gpus(device_id, num_gpus)

    def set_run_total_num_gpus(self, run_id, num_gpus):
        try:
            self.redis_connection.set(self.get_run_total_num_gpus_key(run_id), num_gpus)
        except Exception as e:
            pass

        ComputeGpuDatabase.get_instance().set_run_total_num_gpus(run_id, num_gpus)

    def set_run_device_ids(self, run_id, device_ids):
        try:
            str_device_ids = self.map_list_to_str(device_ids)
            self.redis_connection.set(self.get_run_device_ids_key(run_id), str_device_ids)
        except Exception as e:
            pass

        ComputeGpuDatabase.get_instance().set_run_device_ids(run_id, device_ids)

    def set_edge_model_id_map(self, run_id, edge_id, model_master_device_id, model_slave_device_id):
        try:
            ids_map = f"{edge_id},{model_master_device_id},{model_slave_device_id}"
            self.redis_connection.set(self.get_edge_model_id_map_key(run_id), ids_map)
        except Exception as e:
            pass

        ComputeGpuDatabase.get_instance().set_edge_model_id_map(run_id, edge_id, model_master_device_id, model_slave_device_id)

    def delete_edge_model_id_map(self, run_id):
        try:
            self.redis_connection.delete(self.get_edge_model_id_map_key(run_id))
        except Exception as e:
            logging.error(f"Error deleting edge_model_id_map: {e}, Traceback: {traceback.format_exc()}")

        ComputeGpuDatabase.get_instance().delete_edge_model_id_map(run_id)

    def set_endpoint_run_id_map(self, endpoint_id, run_id):
        try:
            self.redis_connection.set(self.get_endpoint_run_id_map_key(endpoint_id), f"{run_id}")
        except Exception as e:
            pass

        ComputeGpuDatabase.get_instance().set_endpoint_run_id_map(endpoint_id, run_id)

    def delete_endpoint_run_id_map(self, endpoint_id):
        try:
            self.redis_connection.delete(self.get_endpoint_run_id_map_key(endpoint_id))
        except Exception as e:
            logging.error(f"Error deleting endpoint_run_id_map: {e}, Traceback: {traceback.format_exc()}")

        ComputeGpuDatabase.get_instance().delete_endpoint_run_id_map(endpoint_id)

    @staticmethod
    def get_device_run_num_gpus_key(device_id, run_id):
        return f"{ComputeGpuCache.FEDML_GLOBAL_DEVICE_RUN_NUM_GPUS_TAG}{device_id}_{run_id}"

    @staticmethod
    def get_device_run_gpu_ids_key(device_id, run_id):
        return f"{ComputeGpuCache.FEDML_GLOBAL_DEVICE_RUN_GPU_IDS_TAG}{device_id}_{run_id}"

    def get_device_available_gpu_ids_key(self, device_id):
        return f"{ComputeGpuCache.FEDML_GLOBAL_DEVICE_AVAILABLE_GPU_IDS_TAG}{device_id}"

    def get_device_total_num_gpus_key(self, device_id):
        return f"{ComputeGpuCache.FEDML_GLOBAL_DEVICE_TOTAL_NUM_GPUS_TAG}{device_id}"

    @staticmethod
    def get_run_total_num_gpus_key(run_id):
        return f"{ComputeGpuCache.FEDML_GLOBAL_RUN_TOTAL_NUM_GPUS_TAG}{run_id}"

    @staticmethod
    def get_run_device_ids_key(run_id):
        return f"{ComputeGpuCache.FEDML_GLOBAL_RUN_DEVICE_IDS_TAG}{run_id}"

    def get_device_run_lock_key(self, device_id, run_id):
        return f"{ComputeGpuCache.FEDML_DEVICE_RUN_LOCK_TAG}{device_id}_{run_id}"

    def get_device_lock_key(self, device_id):
        return f"{ComputeGpuCache.FEDML_DEVICE_LOCK_TAG}{device_id}"

    def get_run_lock_key(self, run_id):
        return f"{ComputeGpuCache.FEDML_RUN_LOCK_TAG}{run_id}"

    def get_run_info_sync_lock_key(self, run_id):
        return f"{ComputeGpuCache.FEDML_RUN_INFO_SYNC_LOCK_TAG}{run_id}"

    @staticmethod
    def get_edge_model_id_map_key(run_id):
        return f"{ComputeGpuCache.FEDML_EDGE_ID_MODEL_DEVICE_ID_MAP_TAG}{run_id}"

    @staticmethod
    def get_endpoint_run_id_map_key(endpoint_id):
        return f"{ComputeGpuCache.FEDML_GLOBAL_ENDPOINT_RUN_ID_MAP_TAG}{endpoint_id}"

    def map_list_to_str(self, list_obj):
        return ComputeUtils.map_list_to_str(list_obj)

    def map_str_list_to_int_list(self, list_obj):
        return ComputeUtils.map_str_list_to_int_list(list_obj)


