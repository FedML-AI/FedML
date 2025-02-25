import json
import os
import time

from sqlalchemy import Column, String, TEXT, Integer, Float, create_engine, and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from fedml.core.common.singleton import Singleton
from .base_db import FedMLBaseDb
from .compute_utils import ComputeUtils
from ..master.server_constants import ServerConstants

Base = declarative_base()


class ComputeStatusDatabase(Singleton, FedMLBaseDb):
    COMPUTE_STATUS_DB = "compute-status.db"

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_instance():
        return ComputeStatusDatabase()

    def get_job_status(self, run_id):
        self.open_job_db()
        job = self.db_connection.query(FedMLJobStatus). \
            filter(FedMLJobStatus.job_id == f'{run_id}').first()
        if job is None:
            return

        return job.job_status

    def get_device_status_in_job(self, device_id, run_id):
        self.open_job_db()
        device = self.db_connection.query(FedMLDeviceStatusInJob). \
            filter(and_(FedMLDeviceStatusInJob.device_id == f'{device_id}',
                        FedMLDeviceStatusInJob.job_id == f'{run_id}')).first()

        return device.device_status

    def set_job_status(self, run_id, job_status):
        self.open_job_db()
        job = self.db_connection.query(FedMLJobStatus). \
            filter(FedMLJobStatus.job_id == f'{run_id}').first()
        if job is None:
            job = FedMLJobStatus(job_id=run_id, job_status=job_status)
            self.db_connection.add(job)
            self.db_connection.commit()
            return

        if run_id is not None:
            job.job_id = run_id
        if job_status is not None:
            job.job_status = job_status

        self.db_connection.commit()

    def set_device_status_in_job(self, run_id, device_id, status):
        self.open_job_db()
        device = self.db_connection.query(FedMLDeviceStatusInJob). \
            filter(and_(FedMLDeviceStatusInJob.device_id == f'{device_id}',
                        FedMLDeviceStatusInJob.job_id == f'{run_id}')).first()
        if device is None:
            job = FedMLDeviceStatusInJob(job_id=run_id, device_id=device_id, device_status=status)
            self.db_connection.add(job)
            self.db_connection.commit()
            return

        if run_id is not None:
            device.job_id = run_id
        if device_id is not None:
            device.device_id = device_id
        if status is not None:
            device.device_status = status

        self.db_connection.commit()

    def set_database_base_dir(self, database_base_dir):
        self.db_base_dir = database_base_dir
        self.init_db_path()

    def init_db_path(self):
        if self.db_base_dir is None:
            if not os.path.exists(ServerConstants.get_database_dir()):
                os.makedirs(ServerConstants.get_database_dir(), exist_ok=True)
            self.db_base_dir = ServerConstants.get_database_dir()

        self.db_path = os.path.join(self.db_base_dir, ComputeStatusDatabase.COMPUTE_STATUS_DB)

    def create_table(self):
        self.open_job_db()
        try:
            Base.metadata.create_all(self.db_engine, checkfirst=True)
        except Exception as e:
            pass

    def drop_table(self):
        self.open_job_db()
        try:
            Base.metadata.drop_all(self.db_engine, checkfirst=True)
        except Exception as e:
            pass


class FedMLJobStatus(Base):
    __tablename__ = 'job_status'

    id = Column(Integer, primary_key=True)
    job_id = Column(TEXT)
    job_status = Column(TEXT)
    timestamp = Column(Integer)


class FedMLDeviceStatusInJob(Base):
    __tablename__ = 'device_status_in_job'

    id = Column(Integer, primary_key=True)
    job_id = Column(TEXT)
    device_id = Column(TEXT)
    device_status = Column(TEXT)
    timestamp = Column(Integer)
