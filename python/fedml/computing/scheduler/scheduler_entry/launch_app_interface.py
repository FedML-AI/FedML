import logging
import os
import sqlite3
import time
import traceback

from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.core.common.singleton import Singleton


class FedMLLaunchAppDataInterface(Singleton):
    MAX_APPS_LIST_SIZE = 50000

    def __init__(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLLaunchAppDataInterface()

    def open_db(self):
        if not os.path.exists(ClientConstants.get_database_dir()):
            os.makedirs(ClientConstants.get_database_dir(), exist_ok=True)
        db_path = os.path.join(ClientConstants.get_database_dir(), "apps.db")
        self.db_connection = sqlite3.connect(db_path)

    def close_db(self):
        if self.db_connection is not None:
            self.db_connection.close()

    def create_app_table(self):
        self.handle_database_compatibility()

        self.open_db()
        current_cursor = self.db_connection.cursor()
        try:
            current_cursor.execute('''CREATE TABLE IF NOT EXISTS apps
                   (app_id INT PRIMARY KEY NOT NULL,
                   app_name TEXT NOT NULL,
                   app_config TEXT,
                   workspace TEXT,
                   workspace_hash TEXT,
                   app_hash TEXT,
                   client_package_url TEXT,
                   client_package_file TEXT,
                   server_package_url TEXT,
                   server_package_file TEXT,
                   client_diff_url TEXT,
                   client_diff_file TEXT,
                   server_diff_url TEXT,
                   server_diff_file TEXT,
                   model_name TEXT,
                   model_version TEXT,
                   model_url TEXT,
                   updated_time TEXT);''')
            self.db_connection.commit()
        except Exception as e:
            pass
        self.db_connection.close()

    def drop_app_table(self):
        self.open_db()
        current_cursor = self.db_connection.cursor()
        try:
            current_cursor.execute('''DROP TABLE IF EXISTS apps;''')
            self.db_connection.commit()
        except Exception as e:
            logging.info("Process compatibility on the local db.")
        self.db_connection.close()

    def get_current_app_from_db(self):
        app_obj = None

        self.open_db()
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("SELECT *  from apps order by updated_time desc limit(1)")
        for row in results:
            app_obj = FedMLLaunchAppDataInterface(row[0], row[1], row[2], row[3], row[4], row[5],
                                                  row[6], row[7], row[8], row[9], row[10], row[11],
                                                  row[12], row[13], row[14], row[15], row[16], row[17])
            # app_obj.show()
            break

        self.db_connection.close()
        return app_obj

    def get_apps_from_db(self):
        app_list_obj = FedMLLaunchAppListModel()

        self.open_db()
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("SELECT * from apps order by updated_time desc")
        for row in results:
            app_obj = FedMLLaunchAppDataInterface(row[0], row[1], row[2], row[3], row[4], row[5],
                                                  row[6], row[7], row[8], row[9], row[10], row[11],
                                                  row[12], row[13], row[14], row[15], row[16], row[17])
            app_list_obj.app_list.append(app_obj)

            if len(app_list_obj.app_list) > FedMLLaunchAppDataInterface.MAX_APPS_LIST_SIZE:
                break

        self.db_connection.close()
        return app_list_obj

    def get_app_by_id(self, app_id):
        if app_id is None:
            return None

        app_obj = None

        self.open_db()
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("SELECT *  from apps where app_id={};".format(app_id))
        for row in results:
            app_obj = FedMLLaunchAppDataInterface(row[0], row[1], row[2], row[3], row[4], row[5],
                                                  row[6], row[7], row[8], row[9], row[10], row[11],
                                                  row[12], row[13], row[14], row[15], row[16], row[17])
            # app_obj.show()
            break

        self.db_connection.close()
        return app_obj

    def insert_app_to_db(self, app):
        self.open_db()
        current_cursor = self.db_connection.cursor()
        app_query_results = current_cursor.execute("SELECT * from apps where app_id={};".format(app.app_id))
        for row in app_query_results:
            self.db_connection.close()
            self.update_app_to_db(app)
            return

        try:
            current_cursor.execute("INSERT INTO apps (\
                app_id, app_name, app_config, workspace, \
                workspace_hash, app_hash, \
                client_package_url, client_package_file,\
                server_package_url, server_package_file, \
                client_diff_url, client_diff_file,\
                server_diff_url, server_diff_file,\
                model_name, model_version, \
                model_url, updated_time) \
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?, ?, ?, ?)",
                                   (app.app_id, app.app_name, app.app_config, app.workspace,
                                    app.workspace_hash, app.app_hash,
                                    app.client_package_url, app.client_package_file,
                                    app.server_package_url, app.server_package_file,
                                    app.client_diff_url, app.client_diff_file,
                                    app.server_diff_url, app.server_diff_file,
                                    app.model_name, app.model_version,
                                    app.model_url, str(time.time())))
        except Exception as e:
            logging.info("Process apps insertion {}.".format(traceback.format_exc()))
        self.db_connection.commit()
        self.db_connection.close()

    def update_app_to_db(self, app):
        self.open_db()
        current_cursor = self.db_connection.cursor()
        try:
            update_statement = \
                "UPDATE apps set {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}  where app_name={}".format(
                    f",app_config='{app.app_config}'" if app.app_config != "" else "",
                    f",workspace='{app.workspace}'" if app.workspace != "" else "",
                    f",workspace_hash='{app.workspace_hash}'" if app.workspace_hash != "" else "",
                    f",app_hash='{app.app_hash}'" if app.app_hash != "" else "",
                    f",client_package_url={app.client_package_url}" if app.client_package_url != 0 else "",
                    f",client_package_file={app.client_package_file}" if app.client_package_file != 0 else "",
                    f",server_package_url='{app.server_package_url}'" if app.server_package_url != "" else "",
                    f",server_package_file='{app.server_package_file}'" if app.server_package_file != "" else "",
                    f",client_diff_url={app.client_diff_url}" if app.client_diff_url != 0 else "",
                    f",client_diff_file={app.client_diff_file}" if app.client_diff_file != 0 else "",
                    f",server_diff_url='{app.server_diff_url}'" if app.server_diff_url != "" else "",
                    f",server_diff_file='{app.server_diff_file}'" if app.server_diff_file != "" else "",
                    f",model_name={app.model_name}" if app.model_name != -1 else "",
                    f",model_version='{app.model_version}'" if app.model_version != "" else "",
                    f",model_url='{app.model_url}'" if app.model_url != "" else "",
                    ",updated_time='" + str(time.time()) + "'",
                    app.app_name)
            current_cursor.execute(update_statement)
            self.db_connection.commit()
        except Exception as e:
            pass
        self.db_connection.close()

    def handle_database_compatibility(self):
        self.open_db()
        should_alter_old_table = False
        current_cursor = self.db_connection.cursor()
        results = current_cursor.execute("select * from sqlite_master where type='table' and name='apps';")
        for row in results:
            table_statement = str(row[4])
            if table_statement.find("running_json") == -1:
                should_alter_old_table = True

        if should_alter_old_table:
            current_cursor.execute("ALTER TABLE apps ADD running_json TEXT;")
            self.db_connection.commit()
            logging.info("Process compatibility on the local db.")

        self.close_db()


class FedMLLaunchAppDataInterface(object):

    def __init__(self):
        self.app_id = 0
        self.app_name = ""
        self.app_config = ""
        self.workspace = ""
        self.workspace_hash = ""
        self.app_hash = ""
        self.client_package_url = ""
        self.client_package_file = ""
        self.server_package_url = ""
        self.server_package_file = ""
        self.client_diff_url = ""
        self.client_diff_file = ""
        self.server_diff_url = ""
        self.server_diff_file = ""
        self.model_name = ""
        self.model_version = ""
        self.model_url = ""
        self.updated_time = ""

    def __init__(self, app_id, app_name, app_config, workspace,
                 workspace_hash, app_hash,
                 client_package_url, client_package_file,
                 server_package_url, server_package_file,
                 client_diff_url, client_diff_file,
                 server_diff_url, server_diff_file,
                 model_name, model_version, model_url, updated_time):
        self.app_id = app_id
        self.app_name = app_name
        self.app_config = app_config
        self.workspace = workspace
        self.workspace_hash = workspace_hash
        self.app_hash = app_hash
        self.client_package_url = client_package_url
        self.client_package_file = client_package_file
        self.server_package_url = server_package_url
        self.server_package_file = server_package_file
        self.client_diff_url = client_diff_url
        self.client_diff_file = client_diff_file
        self.server_diff_url = server_diff_url
        self.server_diff_file = server_diff_file
        self.model_name = model_name
        self.model_version = model_version
        self.model_url = model_url
        self.updated_time = updated_time

    def show(self):
        logging.info(
            f"App object, app id {self.app_id}, app name {self.app_name}, "
            f"app config {self.app_config}, workspace {self.workspace}, "
            f"client package url {self.client_package_url}, client package file {self.client_package_file}, "
            f"server package url {self.server_package_url}, server package file {self.server_package_file}, "
            f"client diff url {self.client_diff_url}, client diff file {self.client_diff_file}, "
            f"server diff url {self.server_diff_url}, server diff file {self.server_diff_file},"
            f"model name {self.model_name}, model version {self.model_version}, model url {self.model_url}, "
            f"update time {self.updated_time}")


class FedMLLaunchAppListModel(object):

    def __init__(self):
        self.total_num = 0
        self.total_page = 0
        self.page_num = 0
        self.page_size = 0
        self.app_list = list()
