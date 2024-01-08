import json
import os
import time

from sqlalchemy import Column, String, TEXT, Integer, Float, create_engine, and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from fedml.core.common.singleton import Singleton
from .base_db import FedMLBaseDb
from .compute_utils import ComputeUtils

Base = declarative_base()


class ComputeGpuDatabase(Singleton, FedMLBaseDb):
    COMPUTE_GPU_DB = "compute-gpu.db"

    def __init__(self):
        super().__init__()


    @staticmethod
    def get_instance():
        return ComputeGpuDatabase()

    def get_device_run_num_gpus(self, device_id, run_id):
        device_run_num_gpus = 0
        gpu_info_list = self.get_device_run_gpu_info(device_id, run_id)
        for gpu_info in gpu_info_list:
            device_run_num_gpus = gpu_info.num_gpus
            break

        return device_run_num_gpus

    def get_device_run_gpu_ids(self, device_id, run_id):
        device_run_gpu_ids = None
        gpu_info_list = self.get_device_run_gpu_info(device_id, run_id)
        for gpu_info in gpu_info_list:
            device_run_gpu_ids = gpu_info.gpu_ids
            break

        return device_run_gpu_ids

    def get_device_available_gpu_ids(self, device_id):
        device_available_gpu_ids = ""
        gpu_info_list = self.get_device_gpu_info(device_id)
        for gpu_info in gpu_info_list:
            device_available_gpu_ids = gpu_info.available_gpu_ids
            break

        return device_available_gpu_ids

    def get_device_total_num_gpus(self, device_id):
        device_total_num_gpus = 0
        gpu_info_list = self.get_device_gpu_info(device_id)
        for gpu_info in gpu_info_list:
            device_total_num_gpus = gpu_info.total_num_gpus
            break

        return device_total_num_gpus

    def get_run_total_num_gpus(self, run_id):
        run_total_num_gpus = 0
        gpu_info_list = self.get_run_gpu_info(run_id)
        for gpu_info in gpu_info_list:
            run_total_num_gpus = gpu_info.total_num_gpus
            break

        return run_total_num_gpus

    def get_run_device_ids(self, run_id):
        run_device_ids = None
        gpu_info_list = self.get_run_gpu_info(run_id)
        for gpu_info in gpu_info_list:
            run_device_ids = gpu_info.device_ids
            break

        return run_device_ids

    def get_edge_model_id_map(self, run_id):
        edge_id, model_master_device_id, model_slave_device_id = None, None, None
        gpu_info_list = self.get_run_gpu_info(run_id)
        for gpu_info in gpu_info_list:
            edge_id = gpu_info.edge_id
            model_master_device_id = gpu_info.model_master_device_id
            model_slave_device_id = gpu_info.model_slave_device_id
            break

        return edge_id, model_master_device_id, model_slave_device_id

    def get_endpoint_run_id_map(self, endpoint_id):
        run_id = None
        endpoint_run_info_list = self.get_endpoint_run_info(endpoint_id)
        for run_info in endpoint_run_info_list:
            run_id = run_info.run_id
            break

        return run_id

    def set_device_run_num_gpus(self, device_id, run_id, num_gpus):
        self.set_device_run_gpu_info(device_id, run_id, num_gpus=num_gpus)

    def set_device_run_gpu_ids(self, device_id, run_id, gpu_ids):
        str_gpu_ids = ComputeUtils.map_list_to_str(gpu_ids)
        self.set_device_run_gpu_info(device_id, run_id, gpu_ids=str_gpu_ids)

    def set_device_available_gpu_ids(self, device_id, gpu_ids):
        str_gpu_ids = ComputeUtils.map_list_to_str(gpu_ids)
        self.set_device_gpu_info(device_id, available_gpu_ids=str_gpu_ids)

    def set_device_total_num_gpus(self, device_id, num_gpus):
        self.set_device_gpu_info(device_id, total_num_gpus=num_gpus)

    def set_run_total_num_gpus(self, run_id, num_gpus):
        self.set_run_gpu_info(run_id, total_num_gpus=num_gpus)

    def set_run_device_ids(self, run_id, device_ids):
        str_device_ids = ComputeUtils.map_list_to_str(device_ids)
        self.set_run_gpu_info(run_id, device_ids=str_device_ids)

    def set_edge_model_id_map(self, run_id, edge_id, model_master_device_id, model_slave_device_id):
        self.set_run_gpu_info(
            run_id, edge_id=edge_id, model_master_device_id=model_master_device_id,
            model_slave_device_id=model_slave_device_id)

    def set_endpoint_run_id_map(self, endpoint_id, run_id):
        self.set_endpoint_run_info(endpoint_id, run_id)

    def set_database_base_dir(self, database_base_dir):
        self.db_base_dir = database_base_dir
        self.init_db_path()

    def init_db_path(self):
        if self.db_base_dir is None:
            if not os.path.exists(ServerConstants.get_database_dir()):
                os.makedirs(ServerConstants.get_database_dir(), exist_ok=True)
            self.db_base_dir = ServerConstants.get_database_dir()

        self.db_path = os.path.join(self.db_base_dir, ComputeGpuDatabase.COMPUTE_GPU_DB)

    def get_device_run_gpu_info(self, device_id, run_id):
        self.open_job_db()
        gpu_info_list = self.db_connection.query(FedMLDeviceRunGpuInfoModel). \
            filter(and_(FedMLDeviceRunGpuInfoModel.device_id == f'{device_id}',
                        FedMLDeviceRunGpuInfoModel.run_id == f'{run_id}')).all()

        return gpu_info_list

    def set_device_run_gpu_info(self, device_id, run_id, num_gpus=None, gpu_ids=None):
        self.open_job_db()
        gpu_info = self.db_connection.query(FedMLDeviceRunGpuInfoModel). \
            filter(and_(FedMLDeviceRunGpuInfoModel.device_id == f'{device_id}',
                        FedMLDeviceRunGpuInfoModel.run_id == f'{run_id}')).first()
        if gpu_info is None:
            gpu_info = FedMLDeviceRunGpuInfoModel(
                device_id=device_id, run_id=run_id, num_gpus=num_gpus, gpu_ids=gpu_ids)
            self.db_connection.add(gpu_info)
            self.db_connection.commit()
            return

        if num_gpus is not None:
            gpu_info.num_gpus = num_gpus
        if gpu_ids is not None:
            gpu_info.gpu_ids = gpu_ids

        self.db_connection.commit()

    def get_device_gpu_info(self, device_id):
        self.open_job_db()
        gpu_info_list = self.db_connection.query(FedMLDeviceGpuInfoModel). \
            filter(FedMLDeviceGpuInfoModel.device_id == f'{device_id}').all()

        return gpu_info_list

    def set_device_gpu_info(self, device_id, available_gpu_ids=None, total_num_gpus=None):
        self.open_job_db()
        gpu_info = self.db_connection.query(FedMLDeviceGpuInfoModel).filter(
            FedMLDeviceGpuInfoModel.device_id == f'{device_id}').first()
        if gpu_info is None:
            gpu_info = FedMLDeviceGpuInfoModel(
                device_id=device_id, available_gpu_ids=available_gpu_ids, total_num_gpus=total_num_gpus)
            self.db_connection.add(gpu_info)
            self.db_connection.commit()
            return

        if available_gpu_ids is not None:
            gpu_info.available_gpu_ids = available_gpu_ids
        if total_num_gpus is not None:
            gpu_info.total_num_gpus = total_num_gpus

        self.db_connection.commit()

    def get_run_gpu_info(self, run_id):
        self.open_job_db()
        gpu_info_list = self.db_connection.query(FedMLRunGpuInfoModel). \
            filter(FedMLRunGpuInfoModel.run_id == f'{run_id}').all()

        return gpu_info_list

    def set_run_gpu_info(
            self, run_id, device_ids=None, total_num_gpus=None, edge_id=None,
            model_master_device_id=None, model_slave_device_id=None):
        self.open_job_db()
        gpu_info = self.db_connection.query(FedMLRunGpuInfoModel).filter(
            FedMLRunGpuInfoModel.run_id == f'{run_id}').first()
        if gpu_info is None:
            gpu_info = FedMLRunGpuInfoModel(
                run_id=run_id, device_ids=device_ids, total_num_gpus=total_num_gpus,
                edge_id=edge_id, model_master_device_id=model_master_device_id,
                model_slave_device_id=model_slave_device_id)
            self.db_connection.add(gpu_info)
            self.db_connection.commit()
            return

        if device_ids is not None:
            gpu_info.device_ids = device_ids
        if total_num_gpus is not None:
            gpu_info.total_num_gpus = total_num_gpus
        if edge_id is not None:
            gpu_info.edge_id = edge_id
        if model_master_device_id is not None:
            gpu_info.model_master_device_id = model_master_device_id
        if model_slave_device_id is not None:
            gpu_info.model_slave_device_id = model_slave_device_id

        self.db_connection.commit()

    def get_endpoint_run_info(self, endpoint_id):
        self.open_job_db()
        endpoint_run_info_list = self.db_connection.query(FedMLEndpointRunInfoModel). \
            filter(FedMLEndpointRunInfoModel.endpoint_id == f'{endpoint_id}').all()

        return endpoint_run_info_list

    def set_endpoint_run_info(self, endpoint_id, run_id):
        self.open_job_db()
        endpoint_run_info = self.db_connection.query(FedMLEndpointRunInfoModel).filter(
            FedMLEndpointRunInfoModel.endpoint_id == f'{endpoint_id}').first()
        if endpoint_run_info is None:
            endpoint_run_info = FedMLEndpointRunInfoModel(
                endpoint_id=endpoint_id, run_id=run_id)
            self.db_connection.add(endpoint_run_info)
            self.db_connection.commit()
            return

        if endpoint_id is not None:
            endpoint_run_info.endpoint_id = endpoint_id
        if run_id is not None:
            endpoint_run_info.run_id = run_id

        self.db_connection.commit()

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


class FedMLDeviceRunGpuInfoModel(Base):
    __tablename__ = 'device_run_gpu_info'

    id = Column(Integer, primary_key=True)
    device_id = Column(TEXT)
    run_id = Column(TEXT)
    num_gpus = Column(Integer)
    gpu_ids = Column(String)


class FedMLDeviceGpuInfoModel(Base):
    __tablename__ = 'device_gpu_info'

    id = Column(Integer, primary_key=True)
    device_id = Column(TEXT)
    available_gpu_ids = Column(String)
    total_num_gpus = Column(Integer)


class FedMLRunGpuInfoModel(Base):
    __tablename__ = 'run_gpu_info'

    id = Column(Integer, primary_key=True)
    run_id = Column(TEXT)
    total_num_gpus = Column(Integer)
    device_ids = Column(String)
    edge_id = Column(String)
    model_master_device_id = Column(String)
    model_slave_device_id = Column(String)


class FedMLEndpointRunInfoModel(Base):
    __tablename__ = 'endpoint_run_info'

    id = Column(Integer, primary_key=True)
    endpoint_id = Column(TEXT)
    run_id = Column(TEXT)

