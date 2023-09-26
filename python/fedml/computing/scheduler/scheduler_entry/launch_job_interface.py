import logging
import os
import sqlite3
import time
import traceback

from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.core.common.singleton import Singleton
from fedml.computing.scheduler.master.server_constants import ServerConstants


class FedMLLaunchJobDataInterface(Singleton):
    MAX_JOB_LIST_SIZE = 50000
    ERRCODE_JOB_FAILED = 1
    ERRCODE_JOB_KILLED = 2

    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLLaunchJobDataInterface()

    def get_current_job(self):
        current_job = self.get_current_job_from_db()
        return current_job

    def get_history_jobs(self):
        history_job_list = self.get_jobs_from_db()
        return history_job_list

    def save_started_job(self, job_id, edge_id, started_time, status, msg, running_json):
        job_obj = FedMLLaunchJobDataInterface()
        job_obj.job_id = job_id
        job_obj.edge_id = edge_id
        job_obj.started_time = started_time
        job_obj.status = status
        job_obj.error_code = 0
        job_obj.msg = msg
        job_obj.running_json = running_json
        self.insert_job_to_db(job_obj)

    def save_ended_job(self, job_id, edge_id, ended_time, status, msg):
        job_obj = FedMLLaunchJobDataInterface()
        job_obj.job_id = job_id
        job_obj.edge_id = edge_id
        job_obj.ended_time = ended_time
        job_obj.failed_time = "0"
        job_obj.status = status
        job_obj.error_code = 0
        job_obj.msg = msg
        self.update_job_to_db(job_obj)

    def save_failed_job(self, job_id, edge_id, failed_time, status, error_code, msg):
        job_obj = FedMLLaunchJobDataInterface()
        job_obj.job_id = job_id
        job_obj.edge_id = edge_id
        job_obj.failed_time = failed_time
        job_obj.ended_time = failed_time
        job_obj.error_code = error_code
        job_obj.status = status
        job_obj.msg = msg
        self.update_job_to_db(job_obj)

    def save_running_job(self, job_id, edge_id, msg):
        job_obj = FedMLLaunchJobDataInterface()
        job_obj.job_id = job_id
        job_obj.edge_id = edge_id
        job_obj.msg = msg
        self.update_job_to_db(job_obj)

    def save_job_status(self, job_id, edge_id, status, msg):
        job_obj = FedMLLaunchJobDataInterface()
        job_obj.job_id = job_id
        job_obj.edge_id = edge_id
        job_obj.status = status
        job_obj.msg = msg
        self.update_job_to_db(job_obj)

    def save_job(self, run_id, edge_id, status, running_json=None):
        if run_id == 0:
            return

        self.create_job_table()

        if status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_INITIALIZING or \
                status == ServerConstants.MSG_MLOPS_SERVER_STATUS_STARTING:
            self.save_started_job(run_id, edge_id,
                                  time.time(),
                                  status,
                                  status, running_json)
        elif status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FAILED:
            self.save_failed_job(run_id, edge_id,
                                 time.time(),
                                 status,
                                 FedMLLaunchJobDataInterface.ERRCODE_JOB_FAILED,
                                 status)
        elif status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_KILLED:
            self.save_failed_job(run_id, edge_id,
                                 time.time(),
                                 status,
                                 FedMLLaunchJobDataInterface.ERRCODE_JOB_KILLED,
                                 status)
        elif status == ClientConstants.MSG_MLOPS_CLIENT_STATUS_FINISHED:
            self.save_ended_job(run_id, edge_id,
                                time.time(),
                                status,
                                status)
        else:
            self.save_job_status(run_id, edge_id,
                                 status,
                                 status)

    def open_db(self):
        if not os.path.exists(ClientConstants.get_database_dir()):
            os.makedirs(ClientConstants.get_database_dir(), exist_ok=True)
        db_path = os.path.join(ClientConstants.get_database_dir(), "launch.db")
        self.db_connection = sqlite3.connect(db_path)

    def close_db(self):
        if self.db_connection is not None:
            self.db_connection.close()

    def create_job_table(self):
        self.handle_database_compatibility()

        self.open_db()
        current_cursor = self.db_connection.cursor()
        try:
            current_cursor.execute('''CREATE TABLE IF NOT EXISTS jobs
                   (job_id INT PRIMARY KEY NOT NULL,
                   edge_id INT NOT NULL,
                   started_time TEXT NULL,
                   ended_time TEXT,
                   progress FLOAT,
                   ETA FLOAT,
                   status TEXT,
                   failed_time TEXT,
                   error_code INT,
                   msg TEXT,
                   app_name TEXT,
                   model_name TEXT,
                   model_endpoint TEXT,
                   updated_time TEXT,
                   running_json TEXT);''')
            self.db_connection.commit()
        except Exception as e:
            pass
        self.db_connection.close()

    def drop_job_table(self):
        self.open_db()
        current_cursor = self.db_connection.cursor()
        try:
            current_cursor.execute('''DROP TABLE IF EXISTS jobs;''')
            self.db_connection.commit()
        except Exception as e:
            logging.info("Process compatibility on the local db.")
        self.db_connection.close()

    def get_current_job_from_db(self):
        job_obj = None

        self.open_db()
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("SELECT *  from jobs order by updated_time desc limit(1)")
        for row in results:
            job_obj = FedMLLaunchJobDataInterface()
            job_obj.job_id = row[0]
            job_obj.edge_id = row[1]
            job_obj.started_time = row[2]
            job_obj.ended_time = row[3]
            job_obj.status = row[6]
            job_obj.failed_time = row[7]
            job_obj.error_code = row[8]
            job_obj.msg = row[9]
            job_obj.app_name = row[10]
            job_obj.model_name = row[11]
            job_obj.model_endpoint = row[12]
            job_obj.updated_time = row[13]
            job_obj.progress = 0
            job_obj.eta = 0
            job_obj.running_json = row[14]
            # job_obj.show()
            break

        self.db_connection.close()
        return job_obj

    def get_jobs_from_db(self):
        job_list_obj = FedMLLaunchJobListModel()

        self.open_db()
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("SELECT * from jobs order by updated_time desc")
        for row in results:
            job_obj = FedMLLaunchJobDataInterface()
            job_obj.job_id = row[0]
            job_obj.edge_id = row[1]
            job_obj.started_time = row[2]
            job_obj.ended_time = row[3]
            job_obj.status = row[6]
            job_obj.failed_time = row[7]
            job_obj.error_code = row[8]
            job_obj.msg = row[9]
            job_obj.app_name = row[10]
            job_obj.model_name = row[11]
            job_obj.model_endpoint = row[12]
            job_obj.updated_time = row[13]
            job_obj.progress = 0
            job_obj.eta = 0
            job_obj.running_json = row[14]
            job_list_obj.job_list.append(job_obj)

            if len(job_list_obj.job_list) > FedMLLaunchJobDataInterface.MAX_JOB_LIST_SIZE:
                break

        self.db_connection.close()
        return job_list_obj

    def get_job_by_id(self, job_id):
        if job_id is None:
            return None

        job_obj = None

        self.open_db()
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("SELECT *  from jobs where job_id={};".format(job_id))
        for row in results:
            job_obj = FedMLLaunchJobDataInterface()
            job_obj.job_id = row[0]
            job_obj.edge_id = row[1]
            job_obj.started_time = row[2]
            job_obj.ended_time = row[3]
            job_obj.status = row[6]
            job_obj.failed_time = row[7]
            job_obj.error_code = row[8]
            job_obj.msg = row[9]
            job_obj.app_name = row[10]
            job_obj.model_name = row[11]
            job_obj.model_endpoint = row[12]
            job_obj.updated_time = row[13]
            job_obj.progress = 0
            job_obj.eta = 0
            job_obj.running_json = row[14]
            # job_obj.show()
            break

        self.db_connection.close()
        return job_obj

    def insert_job_to_db(self, job):
        self.open_db()
        current_cursor = self.db_connection.cursor()
        job_query_results = current_cursor.execute("SELECT * from jobs where job_id={};".format(job.job_id))
        for row in job_query_results:
            self.db_connection.close()
            self.update_job_to_db(job)
            return

        try:
            current_cursor.execute("INSERT INTO jobs (\
                job_id, edge_id, started_time, ended_time, progress, ETA, status, failed_time, error_code, msg, \
                app_name, model_name, model_endpoint, \
                updated_time, running_json) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                   (job.job_id, job.edge_id, job.started_time, job.ended_time,
                                    job.progress, job.eta, job.status, job.failed_time,
                                    job.error_code, job.msg,
                                    job.app_name, job.model_name, job.model_endpoint,
                                    str(time.time()), job.running_json))
        except Exception as e:
            logging.info("Process jobs insertion {}.".format(traceback.format_exc()))
        self.db_connection.commit()
        self.db_connection.close()

    def update_job_to_db(self, job):
        self.open_db()
        current_cursor = self.db_connection.cursor()
        try:
            update_statement = "UPDATE jobs set {} {} {} {} {} {} {} {} {} {} {} {} {} where job_id={}".format(
                f"edge_id={job.edge_id}" if job.edge_id != 0 else "",
                f",started_time='{job.started_time}'" if job.started_time != "" else "",
                f",ended_time='{job.ended_time}'" if job.ended_time != "" else "",
                f",progress={job.progress}" if job.progress != 0 else "",
                f",eta={job.eta}" if job.eta != 0 else "",
                f",status='{job.status}'" if job.status != "" else "",
                f",failed_time='{job.failed_time}'" if job.failed_time != "" else "",
                f",error_code={job.error_code}" if job.error_code != -1 else "",
                f",msg='{job.msg}'" if job.msg != "" else "",
                f",app_name='{job.app_name}'" if job.app_name != "" else "",
                f",model_name='{job.model_name}'" if job.model_name != "" else "",
                f",model_endpoint='{job.model_endpoint}'" if job.model_endpoint != "" else "",
                ",updated_time='" + str(time.time()) + "'",
                job.job_id)
            current_cursor.execute(update_statement)
            self.db_connection.commit()
        except Exception as e:
            pass
        self.db_connection.close()

    def handle_database_compatibility(self):
        self.open_db()
        should_alter_old_table = False
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("select * from sqlite_master where type='table' and name='jobs';")
        for row in results:
            table_statement = str(row[4])
            if table_statement.find("running_json") == -1:
                should_alter_old_table = True

        if should_alter_old_table:
            current_cursor.execute("ALTER TABLE jobs ADD running_json TEXT;")
            self.db_connection.commit()
            logging.info("Process compatibility on the local db.")

        self.close_db()

    def get_agent_status(self, edge_id=0):
        self.open_db()
        enabled = 1
        current_cursor = self.db_connection.cursor()
        try:
            results = current_cursor.execute("SELECT *  from agent_status where edge_id={};".format(edge_id))
            for row in results:
                out_edge_id = row[0]
                enabled = row[1] > 0
                update_time = row[2]
                break
        except Exception as e:
            pass

        self.db_connection.close()
        return enabled

    def insert_agent_status_to_db(self, agent_status, edge_id=0):
        self.create_agent_status_table()
        self.open_db()
        current_cursor = self.db_connection.cursor()
        query_results = current_cursor.execute("SELECT * from agent_status where edge_id={};".format(edge_id))
        for row in query_results:
            self.db_connection.close()
            self.update_agent_status_to_db(agent_status, edge_id)
            return

        try:
            current_cursor.execute("INSERT INTO agent_status (\
                edge_id, enabled, \
                updated_time) \
                VALUES (?, ?, ?)",
                                   (edge_id, agent_status,
                                    str(time.time())))
        except Exception as e:
            logging.info("Process agent status insertion {}.".format(traceback.format_exc()))
        self.db_connection.commit()
        self.db_connection.close()

    def update_agent_status_to_db(self, agent_status, edge_id=0):
        self.create_agent_status_table()
        self.open_db()
        current_cursor = self.db_connection.cursor()
        try:
            update_statement = "UPDATE agent_status set {} {} {} where edge_id={}".format(
                f"edge_id={edge_id}",
                f",enabled={agent_status}",
                ",updated_time='" + str(time.time()) + "'",
                edge_id)
            current_cursor.execute(update_statement)
            self.db_connection.commit()
        except Exception as e:
            logging.info("update agent status exception {}".format(traceback.format_exc()))
            pass
        self.db_connection.close()


class FedMLLaunchJobDataInterface(object):

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
        self.status = ""
        self.running_json = ""
        self.app_name = ""
        self.model_name = ""
        self.model_endpoint = ""

    def __init__(self, job_id, edge_id, started_time, ended_time, progress, eta,
                 failed_time, error_code, msg, updated_time, status, running_json,
                 app_name, model_name, model_endpoint):
        self.job_id = job_id
        self.edge_id = edge_id
        self.started_time = started_time
        self.ended_time = ended_time
        self.progress = progress
        self.eta = eta
        self.failed_time = failed_time
        self.error_code = error_code
        self.msg = msg
        self.updated_time = updated_time
        self.status = status
        self.running_json = running_json
        self.app_name = app_name
        self.model_name = model_name
        self.model_endpoint = model_endpoint

    def show(self):
        logging.info(
            "Job object, job id {}, edge id {}, started time {},"
            "ended time {}, progress {}, eta {}, status {},"
            "failed time {}, error code {}, msg {},"
            "updated time {}, app name {}, model name {},"
            "model endpoint {}".format(
                self.job_id, self.edge_id, self.started_time,
                self.ended_time, self.progress, self.eta, self.status,
                self.failed_time, self.error_code, self.msg,
                self.updated_time, self.app_name, self.model_name,
                self.model_endpoint
        ))


class FedMLLaunchJobListModel(object):

    def __init__(self):
        self.total_num = 0
        self.total_page = 0
        self.page_num = 0
        self.page_size = 0
        self.job_list = list()
