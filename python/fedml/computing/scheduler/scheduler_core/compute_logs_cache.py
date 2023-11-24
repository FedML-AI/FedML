import json
import math

from .business_models import LogResponseModel


class ComputeLogsCache(object):

    FEDML_RUN_LOG_INDEX_TAG = "FEDML_RUN_LOG_INDEX_TAG-"
    FEDML_RUN_LOG_LIST_TAG = "FEDML_RUN_LOG_LIST_TAG-"
    FEDML_RUN_EDGE_LOG_LIST_TAG = "FEDML_RUN_EDGE_LOG_LIST_TAG-"

    FEDML_RUN_LOGS_GET_SIZE_PER_LOOP = 1000

    def __init__(self, redis_connection):
        self.redis_connection = redis_connection

    def save_run_logs(self, logs_model, run_id=None):
        run_id = logs_model.run_id if run_id is None else run_id
        log_start_index = self.get_run_logs_size(run_id)
        log_list_key = self.get_run_log_list_key(run_id)
        with self.redis_connection.pipeline(transaction=False) as p:
            for log_line in logs_model.log_list:
                p.rpush(log_list_key, log_line)
            p.execute()

        log_end_index = self.get_run_logs_size(run_id)
        self._save_run_log_index(run_id, logs_model, log_start_index, log_end_index)

    def get_run_logs(self, run_id, edge_id=-1, page_num=1, page_size=100):
        if edge_id != -1:
            edge_log_list = self.query_edge_logs_from_run_log_list(run_id, edge_id)
            total_size = len(edge_log_list)
            total_pages = math.ceil(total_size / page_size)
            query_start_index = (page_num-1) * page_size
            query_end_index = query_start_index + page_size
            query_end_index = total_size if query_end_index > total_size else query_end_index
            return LogResponseModel(
                run_id, edge_id, total_pages, total_size, page_num, edge_log_list[query_start_index: query_end_index])

        log_list_key = self.get_run_log_list_key(run_id)
        logs_len = self.get_run_logs_size(run_id)
        total_size = logs_len
        total_pages = math.ceil(total_size / page_size)
        query_start_index = (page_num-1) * page_size
        query_end_index = query_start_index + page_size
        query_end_index = total_size if query_end_index > total_size else query_end_index
        log_list = self.redis_connection.lrange(log_list_key, query_start_index, query_end_index)

        return LogResponseModel(run_id, edge_id, total_pages, total_size, page_num, log_list)

    def query_edge_logs_from_run_log_list(self, run_id, edge_id):
        log_list_key = self.get_run_log_list_key(run_id)
        logs_len = self.get_run_logs_size(run_id)
        index = 0
        result_list = list()
        while index < logs_len:
            log_list = self.redis_connection.lrange(
                log_list_key, index, index+ComputeLogsCache.FEDML_RUN_LOGS_GET_SIZE_PER_LOOP)
            if log_list is None or len(log_list) <= 0:
                break

            for log_line in log_list:
                if str(log_line).find(f"@device-id-{edge_id}") != -1:
                    result_list.append(log_line)

            index += ComputeLogsCache.FEDML_RUN_LOGS_GET_SIZE_PER_LOOP

        return result_list

    def get_run_logs_size(self, run_id):
        log_list_key = self.get_run_log_list_key(run_id)
        logs_len = self.redis_connection.llen(log_list_key)
        return logs_len

    def _save_run_log_index(self, run_id, logs_model, log_start_index, log_end_index):
        run_id = logs_model.run_id if run_id is None else run_id
        log_index_key = self.get_run_log_index_key(run_id)
        log_index_info = {
            "log_list_start_index": log_start_index, "log_list_end_index": log_end_index,
            "run_id": logs_model.run_id, "edge_id": logs_model.edge_id, "create_time": logs_model.create_time,
            "update_time": logs_model.update_time, "created_by": logs_model.created_by, "updated_by": logs_model.updated_by
        }

        self.redis_connection.set(log_index_key, json.dumps(log_index_info))

    def _save_run_edge_log_list(self, run_id, edge_id, log_list):
        edge_log_list_key = self.get_run_edge_log_list_key(run_id, edge_id)
        with self.redis_connection.pipeline(transaction=False) as p:
            for log_line in log_list:
                p.rpush(edge_log_list_key, log_line)
            p.execute()

    def _get_run_edge_log_list(self, run_id, edge_id):
        edge_log_list_key = self.get_run_edge_log_list_key(run_id, edge_id)
        log_list = self.redis_connection.lrange(
            edge_log_list_key, 0, -1)
        return log_list

    def get_run_log_index_key(self, run_id):
        return f"{ComputeLogsCache.FEDML_RUN_LOG_INDEX_TAG}{run_id}"

    def get_run_log_list_key(self, run_id):
        return f"{ComputeLogsCache.FEDML_RUN_LOG_LIST_TAG}{run_id}"

    def get_run_edge_log_list_key(self, run_id, edge_id):
        return f"{ComputeLogsCache.FEDML_RUN_EDGE_LOG_LIST_TAG}{run_id}-{edge_id}"


