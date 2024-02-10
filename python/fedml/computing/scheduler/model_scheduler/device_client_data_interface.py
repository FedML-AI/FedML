import logging
import os
import sqlite3
import time
import traceback

from fedml.computing.scheduler.model_scheduler.device_client_constants import ClientConstants
from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants


class FedMLClientDataInterface(Singleton):
    MAX_JOB_LIST_SIZE = 50000
    ERRCODE_JOB_FAILED = 1
    ERRCODE_JOB_KILLED = 2
    JOBS_DB = "model-jobs.db"

    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLClientDataInterface()

    def get_current_job(self):
        current_job = self.get_current_job_from_db()
        return current_job

    def get_history_jobs(self):
        history_job_list = self.get_jobs_from_db()
        return history_job_list

    def save_started_job(self, job_id, edge_id, started_time, status, msg, running_json):
        job_obj = FedMLClientJobModel()
        job_obj.job_id = job_id
        job_obj.edge_id = edge_id
        job_obj.started_time = started_time
        job_obj.status = status
        job_obj.error_code = 0
        job_obj.msg = msg
        job_obj.running_json = running_json
        self.insert_job_to_db(job_obj)

    def save_ended_job(self, job_id, edge_id, ended_time, status, msg):
        job_obj = FedMLClientJobModel()
        job_obj.job_id = job_id
        job_obj.edge_id = edge_id
        job_obj.ended_time = ended_time
        job_obj.failed_time = "0"
        job_obj.status = status
        job_obj.error_code = 0
        job_obj.msg = msg
        self.update_job_to_db(job_obj)

    def save_failed_job(self, job_id, edge_id, failed_time, status, error_code, msg):
        job_obj = FedMLClientJobModel()
        job_obj.job_id = job_id
        job_obj.edge_id = edge_id
        job_obj.failed_time = failed_time
        job_obj.ended_time = failed_time
        job_obj.error_code = error_code
        job_obj.status = status
        job_obj.msg = msg
        self.update_job_to_db(job_obj)

    def save_running_job(self, job_id, edge_id, round_index, total_rounds, msg):
        job_obj = FedMLClientJobModel()
        job_obj.job_id = job_id
        job_obj.edge_id = edge_id
        job_obj.round_index = round_index
        job_obj.total_rounds = total_rounds
        job_obj.msg = msg
        self.update_job_to_db(job_obj)

    def save_job_status(self, job_id, edge_id, status, msg):
        job_obj = FedMLClientJobModel()
        job_obj.job_id = job_id
        job_obj.edge_id = edge_id
        job_obj.status = status
        job_obj.msg = msg
        self.update_job_to_db(job_obj)

    def save_job_running_json(self, job_id, edge_id, running_json):
        job_obj = FedMLClientJobModel()
        job_obj.job_id = job_id
        job_obj.edge_id = edge_id
        job_obj.running_json = running_json
        self.update_job_to_db(job_obj)

    def save_job(self, run_id, edge_id, status, running_json=None):
        if run_id == 0:
            return

        self.create_job_table()

        if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_INITIALIZING or \
                status == ServerConstants.MSG_MLOPS_SERVER_STATUS_STARTING:
            if self.get_job_by_edge_id_and_job_id(run_id, edge_id) is not None:
                # Update the job status and running json
                self.save_job_status(run_id, edge_id, status, status)
                self.save_job_running_json(run_id, edge_id, running_json)
            else:
                # Insert a new job
                self.save_started_job(run_id, edge_id,
                                      time.time(),
                                      status,
                                      status, running_json)
        elif status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED:
            self.save_failed_job(run_id, edge_id,
                                 time.time(),
                                 status,
                                 FedMLClientDataInterface.ERRCODE_JOB_FAILED,
                                 status)
        elif status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED:
            self.save_failed_job(run_id, edge_id,
                                 time.time(),
                                 status,
                                 FedMLClientDataInterface.ERRCODE_JOB_KILLED,
                                 status)
        elif status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED:
            self.save_ended_job(run_id, edge_id,
                                time.time(),
                                status,
                                status)
        else:
            # Update the job status
            self.save_job_status(run_id, edge_id,
                                 status,
                                 status)

    def save_job_result(self, job_id, edge_id, deployment_result=None):
        self.create_job_table()

        job_obj = FedMLClientJobModel()
        job_obj.job_id = job_id
        job_obj.edge_id = edge_id
        job_obj.deployment_result = deployment_result
        self.update_job_to_db(job_obj)

    def open_job_db(self):
        if not os.path.exists(ClientConstants.get_database_dir()):
            os.makedirs(ClientConstants.get_database_dir(), exist_ok=True)
        job_db_path = os.path.join(ClientConstants.get_database_dir(), FedMLClientDataInterface.JOBS_DB)
        self.db_connection = sqlite3.connect(job_db_path)

    def close_job_db(self):
        if self.db_connection is not None:
            self.db_connection.close()

    def create_job_table(self):
        self.handle_database_compatibility()

        self.open_job_db()
        current_cursor = self.db_connection.cursor()
        try:
            # Job id is not unique, because one job may be executed by different edges node.
            current_cursor.execute('''CREATE TABLE IF NOT EXISTS jobs
                   (job_id INT NOT NULL,
                   edge_id INT NOT NULL,
                   started_time TEXT NULL,
                   ended_time TEXT,
                   progress FLOAT,
                   ETA FLOAT,
                   status TEXT,
                   failed_time TEXT,
                   error_code INT,
                   msg TEXT,
                   updated_time TEXT,
                   round_index INT,
                   total_rounds INT,
                   running_json TEXT,
                   deployment_result TEXT);''')
            self.db_connection.commit()
        except Exception as e:
            pass
        self.db_connection.close()

    def drop_job_table(self):
        self.open_job_db()
        current_cursor = self.db_connection.cursor()
        try:
            current_cursor.execute('''DROP TABLE IF EXISTS jobs;''')
            self.db_connection.commit()
        except Exception as e:
            logging.info("Process compatibility on the local db.")
        self.db_connection.close()

    def get_current_job_from_db(self):
        job_obj = None

        self.open_job_db()
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("SELECT *  from jobs order by updated_time desc limit(1)")
        for row in results:
            job_obj = FedMLClientJobModel()
            job_obj.job_id = row[0]
            job_obj.edge_id = row[1]
            job_obj.started_time = row[2]
            job_obj.ended_time = row[3]
            job_obj.status = row[6]
            job_obj.failed_time = row[7]
            job_obj.error_code = row[8]
            job_obj.msg = row[9]
            job_obj.updated_time = row[10]
            job_obj.round_index = row[11]
            job_obj.total_rounds = row[12]
            job_obj.progress = (0 if job_obj.total_rounds == 0 else job_obj.round_index / job_obj.total_rounds)
            total_time = (0 if job_obj.progress == 0 else (float(job_obj.updated_time) - float(job_obj.started_time))
                                                          / job_obj.progress)
            job_obj.eta = total_time * (1.0 - job_obj.progress)
            job_obj.running_json = row[13]
            job_obj.deployment_result = row[14]
            # job_obj.show()
            break

        self.db_connection.close()
        return job_obj

    def get_jobs_from_db(self):
        job_list_obj = FedMLClientJobListModel()

        self.open_job_db()
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("SELECT * from jobs order by updated_time desc")
        for row in results:
            job_obj = FedMLClientJobModel()
            job_obj.job_id = row[0]
            job_obj.edge_id = row[1]
            job_obj.started_time = row[2]
            job_obj.ended_time = row[3]
            job_obj.status = row[6]
            job_obj.failed_time = row[7]
            job_obj.error_code = row[8]
            job_obj.msg = row[9]
            job_obj.updated_time = row[10]
            job_obj.round_index = row[11]
            job_obj.total_rounds = row[12]
            job_obj.progress = (0 if job_obj.total_rounds == 0 else job_obj.round_index / job_obj.total_rounds)
            total_time = (0 if job_obj.progress == 0 else (float(job_obj.updated_time) - float(job_obj.started_time))
                                                          / job_obj.progress)
            job_obj.eta = total_time * (1.0 - job_obj.progress)
            job_obj.running_json = row[13]
            job_obj.deployment_result = row[14]
            job_list_obj.job_list.append(job_obj)

            if len(job_list_obj.job_list) > FedMLClientDataInterface.MAX_JOB_LIST_SIZE:
                break

        self.db_connection.close()
        return job_list_obj

    def get_job_by_id(self, job_id):
        if job_id is None:
            return None
        job_obj = None

        self.open_job_db()
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("SELECT *  from jobs where job_id={};".format(job_id))
        for row in results:
            job_obj = FedMLClientJobModel()
            job_obj.job_id = row[0]
            job_obj.edge_id = row[1]
            job_obj.started_time = row[2]
            job_obj.ended_time = row[3]
            job_obj.status = row[6]
            job_obj.failed_time = row[7]
            job_obj.error_code = row[8]
            job_obj.msg = row[9]
            job_obj.updated_time = row[10]
            job_obj.round_index = row[11]
            job_obj.total_rounds = row[12]
            job_obj.progress = (0 if job_obj.total_rounds == 0 else job_obj.round_index / job_obj.total_rounds)
            total_time = (0 if job_obj.progress == 0 else (float(job_obj.updated_time) - float(job_obj.started_time))
                                                          / job_obj.progress)
            job_obj.eta = total_time * (1.0 - job_obj.progress)
            job_obj.running_json = row[13]
            job_obj.deployment_result = row[14]
            # job_obj.show()
            break

        self.db_connection.close()
        return job_obj

    def get_job_by_edge_id_and_job_id(self, job_id, edge_id):
        if job_id is None:
            return None
        job_obj = None

        self.open_job_db()
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("SELECT *  from jobs where job_id={} and edge_id={};".format(job_id, edge_id))
        for row in results:
            job_obj = FedMLClientJobModel()
            job_obj.job_id = row[0]
            job_obj.edge_id = row[1]
            job_obj.started_time = row[2]
            job_obj.ended_time = row[3]
            job_obj.status = row[6]
            job_obj.failed_time = row[7]
            job_obj.error_code = row[8]
            job_obj.msg = row[9]
            job_obj.updated_time = row[10]
            job_obj.round_index = row[11]
            job_obj.total_rounds = row[12]
            job_obj.progress = (0 if job_obj.total_rounds == 0 else job_obj.round_index / job_obj.total_rounds)
            total_time = (0 if job_obj.progress == 0 else (float(job_obj.updated_time) - float(job_obj.started_time))
                                                          / job_obj.progress)
            job_obj.eta = total_time * (1.0 - job_obj.progress)
            job_obj.running_json = row[13]
            job_obj.deployment_result = row[14]
            # job_obj.show()
            break

        self.db_connection.close()
        return job_obj

    def insert_job_to_db(self, job):
        self.open_job_db()
        current_cursor = self.db_connection.cursor()

        try:
            current_cursor.execute("INSERT INTO jobs (\
                job_id, edge_id, started_time, ended_time, progress, ETA, status, failed_time, error_code, msg, \
                updated_time, round_index, total_rounds, running_json, deployment_result) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                   (job.job_id, job.edge_id, job.started_time, job.ended_time,
                                    job.progress, job.eta, job.status, job.failed_time,
                                    job.error_code, job.msg, str(time.time()),
                                    job.round_index, job.total_rounds, job.running_json, job.deployment_result))
        except Exception as e:
            logging.info("Process jobs insertion {}.".format(traceback.format_exc()))
        self.db_connection.commit()
        self.db_connection.close()

    def update_job_to_db(self, job):
        self.open_job_db()
        current_cursor = self.db_connection.cursor()
        try:
            update_statement = "UPDATE jobs set {} {} {} {} {} {} {} {} {} {} {} {} {} where job_id={} \
            and edge_id={}".format(
                f"edge_id={job.edge_id}" if job.edge_id != 0 else "",
                f",started_time='{job.started_time}'" if job.started_time != "" else "",
                f",ended_time='{job.ended_time}'" if job.ended_time != "" else "",
                f",progress={job.progress}" if job.progress != 0 else "",
                f",eta={job.eta}" if job.eta != 0 else "",
                f",status='{job.status}'" if job.status != "" else "",
                f",failed_time='{job.failed_time}'" if job.failed_time != "" else "",
                f",error_code={job.error_code}" if job.error_code != -1 else "",
                f",msg='{job.msg}'" if job.msg != "" else "",
                ",updated_time='" + str(time.time()) + "'",
                f",round_index={job.round_index}" if job.round_index != 0 else "",
                f",total_rounds={job.total_rounds}" if job.total_rounds != 0 else "",
                f",deployment_result='{job.deployment_result}'" if job.deployment_result != "" else "",
                job.job_id,
                job.edge_id)
            current_cursor.execute(update_statement)
            self.db_connection.commit()
        except Exception as e:
            pass
        self.db_connection.close()

    def delete_job_from_db(self, job):
        self.open_job_db()
        current_cursor = self.db_connection.cursor()
        try:
            delete_statement = f"DELETE from jobs where job_id={job}"
            current_cursor.execute(delete_statement)
            self.db_connection.commit()
        except Exception as e:
            pass
        self.db_connection.close()

    def handle_database_compatibility(self):
        self.open_job_db()
        should_alter_old_table = False
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("select * from sqlite_master where type='table' and name='jobs';")
        for row in results:
            table_statement = str(row[4])
            if table_statement.find("deployment_result") == -1:
                should_alter_old_table = True

        if should_alter_old_table:
            current_cursor.execute("ALTER TABLE jobs ADD deployment_result TEXT;")
            self.db_connection.commit()
            logging.info("Process compatibility on the local db.")

        self.close_job_db()


class FedMLClientJobModel(object):

    def __init__(self):
        self.job_id = 0
        self.edge_id = 0
        self.started_time = ""
        self.ended_time = ""
        self.progress = 0
        self.eta = 0
        self.failed_time = ""
        self.error_code = -1
        self.msg = ""
        self.updated_time = ""
        self.round_index = 0
        self.total_rounds = 0
        self.status = ""
        self.running_json = ""
        self.deployment_result = ""

    def show(self):
        logging.info("Job object, job id {}, edge id {}, started time {},"
                     "ended time {}, progress {}, eta {}, status {},"
                     "failed time {}, error code {}, msg {},"
                     "updated time {}, round index {}, total rounds {}".format(
            self.job_id, self.edge_id, self.started_time,
            self.ended_time, self.progress, self.eta, self.status,
            self.failed_time, self.error_code, self.msg,
            self.updated_time, self.round_index, self.total_rounds
        ))


class FedMLClientJobListModel(object):

    def __init__(self):
        self.total_num = 0
        self.total_page = 0
        self.page_num = 0
        self.page_size = 0
        self.job_list = list()
