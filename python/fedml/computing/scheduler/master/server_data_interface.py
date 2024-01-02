
import os
import sqlite3

from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.slave.client_data_interface import FedMLClientDataInterface


class FedMLServerDataInterface(FedMLClientDataInterface):
    MAX_JOB_LIST_SIZE = 50000
    ERRCODE_JOB_FAILED = 1
    ERRCODE_JOB_KILLED = 2

    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLServerDataInterface()

    def open_job_db(self):
        if not os.path.exists(ServerConstants.get_database_dir()):
            os.makedirs(ServerConstants.get_database_dir(), exist_ok=True)
        job_db_path = os.path.join(ServerConstants.get_database_dir(), "jobs.db")
        self.db_connection = sqlite3.connect(job_db_path)


