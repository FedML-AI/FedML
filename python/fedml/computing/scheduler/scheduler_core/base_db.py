import json
import os
import time

from sqlalchemy import Column, String, TEXT, Integer, Float, create_engine, and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from fedml.core.common.singleton import Singleton


class FedMLBaseDb:
    COMPUTE_GPU_DB = "compute-gpu.db"

    def __init__(self):
        if not hasattr(self, "db_connection"):
            self.db_connection = None
        if not hasattr(self, "db_engine"):
            self.db_engine = None
        if not hasattr(self, "db_base_dir"):
            self.db_base_dir = None
        if not hasattr(self, "db_path"):
            self.db_path = None

    def open_job_db(self):
        if self.db_connection is not None:
            return

        self.db_engine = create_engine('sqlite:////{}'.format(self.db_path), echo=False)

        db_session_class = sessionmaker(bind=self.db_engine)
        self.db_connection = db_session_class()

    def close_job_db(self):
        if self.db_connection is not None:
            self.db_connection.close()

