
import os
import sqlite3

from fedml.cli.model_deployment.device_server_constants import ServerConstants
from fedml.cli.model_deployment.device_client_data_interface import FedMLClientDataInterface


class FedMLServerDataInterface(FedMLClientDataInterface):
    MAX_JOB_LIST_SIZE = 50000
    ERRCODE_JOB_FAILED = 1
    ERRCODE_JOB_KILLED = 2
    JOBS_DB = "model-jobs.db"

    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLServerDataInterface()

    def open_job_db(self):
        if not os.path.exists(ServerConstants.get_database_dir()):
            os.makedirs(ServerConstants.get_database_dir(), exist_ok=True)
        job_db_path = os.path.join(ServerConstants.get_database_dir(), FedMLServerDataInterface.JOBS_DB)
        self.db_connection = sqlite3.connect(job_db_path)


