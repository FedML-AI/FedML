import logging
import traceback
from multiprocessing import Process

from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.scheduler_core.compute_cache_manager import ComputeCacheManager
from fedml.computing.scheduler.slave import client_data_interface
from fedml.computing.scheduler.model_scheduler import device_client_data_interface
from fedml.core.common.singleton import Singleton

from .job_utils import JobRunnerUtils
from ..model_scheduler.device_model_cache import FedMLModelCache
from ..slave import client_constants
from ..model_scheduler import device_client_constants


class JobCleanup(Singleton):
    def __init__(self):
        if not hasattr(self, "sync_data_proc"):
            self.sync_data_proc = None

    @staticmethod
    def get_instance():
        return JobCleanup()

    def sync_data_on_startup(self, edge_id):
        if self.sync_data_proc is None:
            self.sync_data_proc = Process(target=JobCleanup.sync_proc, args=(edge_id,))
            self.sync_data_proc.start()

    @staticmethod
    def sync_proc(edge_id):
        JobRunnerUtils.get_instance().reset_available_gpu_id_list(edge_id)
        JobCleanup.get_instance().sync_run_process_gpu()
        JobCleanup.get_instance().sync_endpoint_process_gpu()

    def sync_run_process_gpu(self):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            with ComputeCacheManager.get_instance().get_redis_connection().lock(
                    ComputeCacheManager.get_instance().get_gpu_cache().get_run_info_sync_lock_key("")
            ):
                count = 0
                job_list = client_data_interface.FedMLClientDataInterface.get_instance().get_jobs_from_db()
                for job in job_list.job_list:
                    count += 1
                    if count >= 1000:
                        break

                    run_process_list = client_constants.ClientConstants.get_learning_process_list(job.job_id)
                    all_run_processes_exited = True if len(run_process_list) <= 0 else False

                    if SchedulerConstants.is_run_completed(job.status):
                        client_constants.ClientConstants.cleanup_learning_process(job.job_id)
                        all_run_processes_exited = True

                    if all_run_processes_exited:
                        JobRunnerUtils.get_instance().release_gpu_ids(job.job_id, job.edge_id)
        except Exception as e:
            logging.info(f"Exception when syncing run process.{traceback.format_exc()}")
            pass

    def sync_endpoint_process_gpu(self):
        try:
            ComputeCacheManager.get_instance().set_redis_params()
            with ComputeCacheManager.get_instance().get_redis_connection().lock(
                    ComputeCacheManager.get_instance().get_gpu_cache().get_run_info_sync_lock_key("")
            ):
                count = 0
                FedMLModelCache.get_instance().set_redis_params()
                try:
                    device_client_data_interface.FedMLClientDataInterface.get_instance().create_job_table()
                except Exception as e:
                    pass
                job_list = device_client_data_interface.FedMLClientDataInterface.get_instance().get_jobs_from_db()
                for job in job_list.job_list:
                    count += 1
                    if count >= 1000:
                        break
                    if job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED or \
                            job.status == device_client_constants.ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED:
                        JobRunnerUtils.get_instance().release_gpu_ids(job.job_id, job.edge_id)
        except Exception as e:
            logging.info("Exception when syncing endpoint process.")
            pass
