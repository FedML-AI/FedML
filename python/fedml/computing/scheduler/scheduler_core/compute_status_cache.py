import logging
import traceback
from .compute_status_db import ComputeStatusDatabase
from ..master.server_constants import ServerConstants


class ComputeStatusCache(object):
    FEDML_JOB_STATUS_TAG = "FEDML_JOB_STATUS_TAG-"
    FEDML_DEVICE_STATUS_IN_JOB_TAG = "FEDML_DEVICE_STATUS_IN_JOB_TAG-"

    def __init__(self, redis_connection):
        self.redis_connection = redis_connection
        ComputeStatusDatabase.get_instance().set_database_base_dir(ServerConstants.get_database_dir())
        ComputeStatusDatabase.get_instance().create_table()

    def save_job_status(self, run_id, status):
        try:
            self.redis_connection.set(self._get_job_status_key(run_id), status)
        except Exception as e:
            logging.error(f"Error setting job status: {e}, Traceback: {traceback.format_exc()}")
            pass

        ComputeStatusDatabase.get_instance().set_job_status(run_id, status)

    def get_job_status(self, run_id):
        status = None
        try:
            if self.redis_connection.exists(self._get_job_status_key(run_id)):
                status = self.redis_connection.get(self._get_job_status_key(run_id))
        except Exception as e:
            logging.error(f"Error getting job status: {e}, Traceback: {traceback.format_exc()}")
            pass

        if status is None:
            status = ComputeStatusDatabase.get_instance().get_job_status(run_id)
            try:
                if status is not None:
                    self.redis_connection.set(self._get_job_status_key(run_id), status)
            except Exception as e:
                pass

        return status

    def save_device_status_in_job(self, run_id, device_id, status):
        if status is None:
            return
        try:
            self.redis_connection.set(self._get_device_status_in_job_key(run_id, device_id), status)
        except Exception as e:
            logging.error(f"Error setting device status in job: {e}, Traceback: {traceback.format_exc()}")
            pass

        ComputeStatusDatabase.get_instance().set_device_status_in_job(run_id, device_id, status)

    def get_device_status_in_job(self, run_id, device_id):
        status = None
        try:
            if self.redis_connection.exists(self._get_device_status_in_job_key(run_id, device_id)):
                status = self.redis_connection.get(self._get_device_status_in_job_key(run_id, device_id))
        except Exception as e:
            logging.error(f"Error getting device status in job: {e}, Traceback: {traceback.format_exc()}")
            pass

        if status is None:
            status = ComputeStatusDatabase.get_instance().get_device_status_in_job(run_id, device_id)
            try:
                if status is not None:
                    self.redis_connection.set(self._get_device_status_in_job_key(run_id, device_id), status)
            except Exception as e:
                pass

        return status

    def _get_job_status_key(self, run_id):
        return f"{ComputeStatusCache.FEDML_JOB_STATUS_TAG}{run_id}"

    def _get_device_status_in_job_key(self, run_id, device_id):
        return f"{ComputeStatusCache.FEDML_DEVICE_STATUS_IN_JOB_TAG}{run_id}-{device_id}"
