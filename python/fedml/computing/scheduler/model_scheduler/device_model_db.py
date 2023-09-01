import json
import os
import time

from fedml.computing.scheduler.model_scheduler.device_server_constants import ServerConstants
from sqlalchemy import Column, String, TEXT, Integer, Float, create_engine, and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class FedMLModelDatabase(object):
    MODEL_DEPLOYMENT_DB = "model-deployment.db"

    def __init__(self):
        self.db_connection = None
        self.db_engine = None

    def __new__(cls, *args, **kw):
        if not hasattr(cls, "_instance"):
            orig = super(FedMLModelDatabase, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
            cls._instance.init()
        return cls._instance

    def init(self):
        pass

    @staticmethod
    def get_instance():
        return FedMLModelDatabase()

    def set_deployment_result(self, end_point_id, end_point_name, model_name, model_version,
                              device_id, deployment_result):
        self.set_deployment_results_info(end_point_id, end_point_name, model_name, model_version,
                                         device_id, deployment_result=deployment_result)

    def set_deployment_status(self, end_point_id, end_point_name, model_name, model_version,
                              device_id, deployment_status):
        self.set_deployment_results_info(end_point_id, end_point_name, model_name, model_version,
                                         device_id, deployment_status=deployment_status)

    def get_deployment_result_list(self, end_point_name, model_name, model_version=None):
        result_list = self.get_deployment_results_info(end_point_name, model_name, model_version)
        ret_result_list = list()
        for result in result_list:
            result_dict = {"cache_device_id": result.device_id, "result": result.deployment_result}
            ret_result_list.append(json.dumps(result_dict))
        return ret_result_list

    def get_deployment_status_list(self, end_point_name, model_name, model_version=None):
        result_list = self.get_deployment_results_info(end_point_name, model_name, model_version)
        ret_status_list = list()
        for result in result_list:
            status_dict = {"cache_device_id": result.device_id, "status": result.deployment_status}
            ret_status_list.append(json.dumps(status_dict))
        return ret_status_list

    def set_end_point_status(self, end_point_id, end_point_name, status):
        self.set_deployment_run_info(end_point_id, end_point_name, end_point_status=status)

    def get_end_point_status(self, end_point_id):
        run_info = self.get_deployment_run_info(end_point_id)
        if run_info is not None:
            return run_info.end_point_status
        return None

    def set_end_point_activation(self, end_point_id, end_point_name, activate_status):
        self.set_deployment_run_info(end_point_id, end_point_name, activated=activate_status)

    def get_end_point_activation(self, end_point_id):
        run_info = self.get_deployment_run_info(end_point_id)
        if run_info is not None:
            return run_info.activated
        return 0

    def set_end_point_device_info(self, end_point_id, end_point_name, device_info):
        self.set_deployment_run_info(end_point_id, end_point_name, device_info=device_info)

    def get_end_point_device_info(self, end_point_id):
        run_info = self.get_deployment_run_info(end_point_id)
        if run_info is not None:
            return run_info.device_info
        return None

    def set_end_point_token(self, end_point_id, end_point_name, model_name, token):
        if self.get_end_point_token(end_point_name, model_name) is not None:
            return
        self.set_deployment_auth_info(end_point_id, end_point_name, model_name, token)

    def get_end_point_token(self, end_point_name, model_name):
        auth_info = self.get_deployment_auth_info(end_point_name, model_name)
        if auth_info is not None:
            return auth_info.token
        return None

    def set_monitor_metrics(self, end_point_id, end_point_name,
                            model_name, model_version,
                            total_latency, avg_latency,
                            total_request_num, current_qps,
                            avg_qps, timestamp, device_id):
        self.set_end_point_metrics(end_point_id, end_point_name,
                                   model_name, model_version,
                                   total_latency=total_latency, avg_latency=avg_latency,
                                   total_request_num=total_request_num, current_qps=current_qps,
                                   avg_qps=avg_qps, timestamp=timestamp, device_id=device_id)

    def get_latest_monitor_metrics(self, end_point_name, model_name, model_version):
        endpoint_metrics = self.get_latest_end_point_metrics(end_point_name, model_name, model_version)
        if endpoint_metrics is None:
            return None
        metrics_dict = {"total_latency": endpoint_metrics.total_latency, "avg_latency": endpoint_metrics.avg_latency,
                        "total_request_num": endpoint_metrics.total_request_num,
                        "current_qps": endpoint_metrics.current_qps,
                        "avg_qps": endpoint_metrics.avg_qps, "timestamp": endpoint_metrics.timestamp,
                        "device_id": endpoint_metrics.device_id}
        return json.dumps(metrics_dict)

    def get_monitor_metrics_item(self, end_point_name, model_name, model_version, index):
        endpoint_metrics = self.get_end_point_metrics_by_index(end_point_name, model_name, model_version, index)
        if endpoint_metrics is None:
            return None
        metrics_dict = {"total_latency": endpoint_metrics.total_latency, "avg_latency": endpoint_metrics.avg_latency,
                        "total_request_num": endpoint_metrics.total_request_num,
                        "current_qps": endpoint_metrics.current_qps,
                        "avg_qps": endpoint_metrics.avg_qps, "timestamp": endpoint_metrics.timestamp,
                        "device_id": endpoint_metrics.device_id}
        return json.dumps(metrics_dict)

    def open_job_db(self):
        if self.db_connection is not None:
            return

        if not os.path.exists(ServerConstants.get_database_dir()):
            os.makedirs(ServerConstants.get_database_dir(), exist_ok=True)
        job_db_path = os.path.join(ServerConstants.get_database_dir(), FedMLModelDatabase.MODEL_DEPLOYMENT_DB)
        self.db_engine = create_engine('sqlite:////{}'.format(job_db_path), echo=False)

        db_session_class = sessionmaker(bind=self.db_engine)
        self.db_connection = db_session_class()

    def close_job_db(self):
        if self.db_connection is not None:
            self.db_connection.close()

    def create_table(self):
        self.open_job_db()
        Base.metadata.create_all(self.db_engine, checkfirst=True)

    def drop_table(self):
        self.open_job_db()
        Base.metadata.drop_all(self.db_engine, checkfirst=True)

    def get_deployment_results_info(self, end_point_name, model_name, model_version):
        self.open_job_db()
        if model_version is None:
            result_info = self.db_connection.query(FedMLDeploymentResultInfoModel). \
                filter(and_(FedMLDeploymentResultInfoModel.end_point_name == f'{end_point_name}',
                            FedMLDeploymentResultInfoModel.model_name == f'{model_name}')).all()
        else:
            result_info = self.db_connection.query(FedMLDeploymentResultInfoModel). \
                filter(and_(FedMLDeploymentResultInfoModel.end_point_name == f'{end_point_name}',
                            FedMLDeploymentResultInfoModel.model_name == f'{model_name}',
                            FedMLDeploymentResultInfoModel.model_version == f'{model_version}')).all()
        return result_info

    def set_deployment_results_info(self, end_point_id, end_point_name,
                                    model_name, model_version, device_id,
                                    deployment_result=None, deployment_status=None):
        self.open_job_db()
        result_info = self.db_connection.query(FedMLDeploymentResultInfoModel). \
            filter(and_(FedMLDeploymentResultInfoModel.end_point_name == f'{end_point_name}',
                        FedMLDeploymentResultInfoModel.model_name == f'{model_name}',
                        FedMLDeploymentResultInfoModel.model_version == f'{model_version}')).first()
        if result_info is None:
            result_info = FedMLDeploymentResultInfoModel(end_point_id=end_point_id,
                                                         end_point_name=end_point_name,
                                                         model_name=model_name,
                                                         model_version=model_version,
                                                         device_id=device_id,
                                                         deployment_result=deployment_result,
                                                         deployment_status=deployment_status)
            self.db_connection.add(result_info)
            self.db_connection.commit()
            return

        if deployment_result is not None:
            result_info.deployment_result = deployment_result
        if deployment_status is not None:
            result_info.deployment_status = deployment_status

        self.db_connection.commit()

    def get_deployment_run_info(self, end_point_id):
        self.open_job_db()
        run_info = self.db_connection.query(FedMLDeploymentRunInfoModel). \
            filter_by(end_point_id=f'{end_point_id}').first()
        return run_info

    def set_deployment_run_info(self, end_point_id, end_point_name,
                                end_point_status=None, device_info=None,
                                activated=None, token=None):
        self.open_job_db()
        run_info = self.db_connection.query(FedMLDeploymentRunInfoModel). \
            filter_by(end_point_id=f'{end_point_id}').first()
        if run_info is None:
            run_info = FedMLDeploymentRunInfoModel(end_point_id=end_point_id,
                                                   end_point_name=end_point_name,
                                                   end_point_status=end_point_status,
                                                   device_info=device_info,
                                                   activated=activated, token=token)
            self.db_connection.add(run_info)
            self.db_connection.commit()
            return

        if end_point_status is not None:
            run_info.end_point_status = end_point_status
        if device_info is not None:
            run_info.device_info = device_info
        if activated is not None:
            run_info.activated = activated
        if token is not None:
            run_info.token = token

        self.db_connection.commit()

    def get_deployment_auth_info(self, end_point_name, model_name):
        self.open_job_db()
        run_info = self.db_connection.query(FedMLDeploymentAuthInfoModel). \
            filter(and_(FedMLDeploymentAuthInfoModel.end_point_name == f'{end_point_name}',
                        FedMLDeploymentAuthInfoModel.model_name == f'{model_name}')).first()
        return run_info

    def set_deployment_auth_info(self, end_point_id, end_point_name, model_name, token):
        self.open_job_db()
        auth_info = self.db_connection.query(FedMLDeploymentAuthInfoModel). \
            filter(and_(FedMLDeploymentAuthInfoModel.end_point_name == f'{end_point_name}',
                        FedMLDeploymentAuthInfoModel.model_name == f'{model_name}')).first()
        if auth_info is None:
            auth_info = FedMLDeploymentAuthInfoModel(end_point_id=end_point_id,
                                                     end_point_name=end_point_name,
                                                     model_name=model_name, token=token)
            self.db_connection.add(auth_info)
            self.db_connection.commit()
            return

        if token is not None:
            auth_info.token = token

        self.db_connection.commit()

    def get_latest_end_point_metrics(self, end_point_name, model_name, model_version):
        self.open_job_db()
        endpoint_metric = self.db_connection.query(FedMLEndPointMetricsModel). \
            filter(and_(FedMLEndPointMetricsModel.end_point_name == f'{end_point_name}',
                        FedMLEndPointMetricsModel.model_name == f'{model_name}',
                        FedMLEndPointMetricsModel.model_version == f'{model_version}')).all()
        if len(endpoint_metric) >= 1:
            return endpoint_metric[-1]
        return None

    def get_end_point_metrics_by_index(self, end_point_name, model_name, model_version, index):
        self.open_job_db()
        endpoint_metric = self.db_connection.query(FedMLEndPointMetricsModel). \
            filter(and_(FedMLEndPointMetricsModel.end_point_name == f'{end_point_name}',
                        FedMLEndPointMetricsModel.model_name == f'{model_name}',
                        FedMLEndPointMetricsModel.model_version == f'{model_version}')). \
            offset(index).limit(1).first()
        return endpoint_metric

    def set_end_point_metrics(self, end_point_id, end_point_name,
                              model_name, model_version,
                              total_latency=None, avg_latency=None,
                              total_request_num=None, current_qps=None,
                              avg_qps=None, timestamp=None, device_id=None):
        self.open_job_db()
        endpoint_metric = self.db_connection.query(FedMLEndPointMetricsModel). \
            filter(and_(FedMLEndPointMetricsModel.end_point_name == f'{end_point_name}',
                        FedMLEndPointMetricsModel.model_name == f'{model_name}',
                        FedMLEndPointMetricsModel.model_version == f'{model_version}')).first()
        if endpoint_metric is None:
            endpoint_metric = FedMLEndPointMetricsModel(end_point_id=end_point_id,
                                                        end_point_name=end_point_name,
                                                        model_name=model_name, model_version=model_version,
                                                        total_latency=total_latency, avg_latency=avg_latency,
                                                        total_request_num=total_request_num, current_qps=current_qps,
                                                        avg_qps=avg_qps, timestamp=timestamp, device_id=device_id)
            self.db_connection.add(endpoint_metric)
            self.db_connection.commit()
            return

        if total_latency is not None:
            endpoint_metric.total_latency = total_latency
        if avg_latency is not None:
            endpoint_metric.avg_latency = avg_latency
        if total_request_num is not None:
            endpoint_metric.total_request_num = total_request_num
        if current_qps is not None:
            endpoint_metric.current_qps = current_qps
        if avg_qps is not None:
            endpoint_metric.avg_qps = avg_qps
        if timestamp is not None:
            endpoint_metric.timestamp = timestamp
        if device_id is not None:
            endpoint_metric.device_id = device_id

        self.db_connection.commit()


class FedMLDeploymentResultInfoModel(Base):
    __tablename__ = 'deployment_result_info'

    id = Column(Integer, primary_key=True)
    end_point_id = Column(TEXT)
    end_point_name = Column(TEXT)
    model_name = Column(TEXT)
    model_version = Column(TEXT)
    device_id = Column(TEXT)
    deployment_result = Column(TEXT)
    deployment_status = Column(TEXT)


class FedMLDeploymentRunInfoModel(Base):
    __tablename__ = 'deployment_run_info'

    id = Column(Integer, primary_key=True)
    end_point_id = Column(TEXT)
    end_point_name = Column(TEXT)
    end_point_status = Column(TEXT)
    device_info = Column(Integer)
    activated = Column(Integer)
    token = Column(TEXT)


class FedMLDeploymentAuthInfoModel(Base):
    __tablename__ = 'deployment_auth_info'

    id = Column(Integer, primary_key=True)
    end_point_id = Column(TEXT)
    end_point_name = Column(TEXT)
    model_name = Column(TEXT)
    token = Column(TEXT)


class FedMLEndPointMetricsModel(Base):
    __tablename__ = 'end_point_metrics'

    id = Column(Integer, primary_key=True)
    end_point_id = Column(TEXT)
    end_point_name = Column(TEXT)
    model_name = Column(TEXT)
    model_version = Column(TEXT)
    total_latency = Column(Float)
    avg_latency = Column(Float)
    total_request_num = Column(Integer)
    current_qps = Column(Float)
    avg_qps = Column(Float)
    timestamp = Column(Integer)
    device_id = Column(TEXT)


def test_deployment_result():
    test_end_point_id = "545"
    test_end_point_name = "EndPoint-98f9f598-5ac7-40a4-b780-54c20f19acaa"
    test_model_name = "alex-test-model"
    test_model_version = "v0-Mon Apr 10 12:30:55 CST 2023"
    test_device_id = "178076"
    test_deployment_result = {"end_point_id": test_end_point_id}
    FedMLModelDatabase.get_instance().create_table()
    FedMLModelDatabase.get_instance().set_deployment_result(test_end_point_id,
                                                            test_end_point_name,
                                                            test_model_name,
                                                            test_model_version,
                                                            test_device_id,
                                                            json.dumps(test_deployment_result))

    result_list = FedMLModelDatabase.get_instance().get_deployment_result_list(test_end_point_name,
                                                                               test_model_name,
                                                                               test_model_version)
    if result_list is None or len(result_list) != 1:
        print("Failed to test for setting and getting deployment result")
    else:
        result_info = json.loads(result_list[0])
        if result_info["cache_device_id"] == test_device_id and \
                result_info["result"] == json.dumps(test_deployment_result):
            print("Succeeded to test for setting and getting deployment result")
        else:
            print("Failed to test for setting and getting deployment result")


def test_deployment_status():
    test_end_point_id = "545"
    test_end_point_name = "EndPoint-98f9f598-5ac7-40a4-b780-54c20f19acaa"
    test_model_name = "alex-test-model"
    test_model_version = "v0-Mon Apr 10 12:30:55 CST 2023"
    test_device_id = "178076"
    test_deployment_status = {"end_point_id": test_end_point_id}
    FedMLModelDatabase.get_instance().create_table()
    FedMLModelDatabase.get_instance().set_deployment_status(test_end_point_id,
                                                            test_end_point_name,
                                                            test_model_name,
                                                            test_model_version,
                                                            test_device_id,
                                                            json.dumps(test_deployment_status))

    status_list = FedMLModelDatabase.get_instance().get_deployment_result_list(test_end_point_name,
                                                                               test_model_name,
                                                                               test_model_version)
    if status_list is None or len(status_list) != 1:
        print("Failed to test for setting and getting deployment status")
    else:
        status_info = json.loads(status_list[0])
        if status_info["cache_device_id"] == test_device_id and \
                status_info["result"] == json.dumps(test_deployment_status):
            print("Succeeded to test for setting and getting deployment status")
        else:
            print("Failed to test for setting and getting deployment status")


def test_end_point_status():
    test_end_point_id = "545"
    test_end_point_name = "EndPoint-98f9f598-5ac7-40a4-b780-54c20f19acaa"
    test_end_point_status = "DEPLOYED"
    FedMLModelDatabase.get_instance().set_end_point_status(test_end_point_id,
                                                           test_end_point_name,
                                                           test_end_point_status)

    status = FedMLModelDatabase.get_instance().get_end_point_status(test_end_point_id)
    if status is None or status != test_end_point_status:
        print("Failed to test for setting and getting end point status")
    else:
        print("Succeeded to test for setting and getting end point status")


def test_end_point_activation():
    test_end_point_id = "545"
    test_end_point_name = "EndPoint-98f9f598-5ac7-40a4-b780-54c20f19acaa"
    test_end_point_activation = 1
    FedMLModelDatabase.get_instance().set_end_point_activation(test_end_point_id,
                                                               test_end_point_name,
                                                               test_end_point_activation)

    activation = FedMLModelDatabase.get_instance().get_end_point_activation(test_end_point_id)
    if activation != test_end_point_activation:
        print("Failed to test for setting and getting end point activation")
    else:
        print("Succeeded to test for setting and getting end point activation")


def test_end_point_device_info():
    test_end_point_id = "545"
    test_end_point_name = "EndPoint-98f9f598-5ac7-40a4-b780-54c20f19acaa"
    test_end_point_device_info = {"device_id": 1}
    FedMLModelDatabase.get_instance().set_end_point_device_info(test_end_point_id,
                                                                test_end_point_name,
                                                                json.dumps(test_end_point_device_info))

    ret_device_info = FedMLModelDatabase.get_instance().get_end_point_device_info(test_end_point_id)
    if ret_device_info != json.dumps(test_end_point_device_info):
        print("Failed to test for setting and getting end point device info")
    else:
        print("Succeeded to test for setting and getting end point device info")


def test_end_point_token():
    test_end_point_id = "545"
    test_end_point_name = "EndPoint-98f9f598-5ac7-40a4-b780-54c20f19acaa"
    test_model_name = "alex-test-model"
    test_end_point_token = "addee88adf9899asdfsdfsfdee"
    FedMLModelDatabase.get_instance().set_end_point_token(test_end_point_id,
                                                          test_end_point_name,
                                                          test_model_name,
                                                          test_end_point_token)

    ret_token = FedMLModelDatabase.get_instance().get_end_point_token(test_end_point_name, test_model_name)
    if ret_token is None or ret_token != test_end_point_token:
        print("Failed to test for setting and getting end point token")
    else:
        print("Succeeded to test for setting and getting end point token")


def test_end_point_monitor_metrics():
    test_end_point_id = "545"
    test_end_point_name = "EndPoint-98f9f598-5ac7-40a4-b780-54c20f19acaa"
    test_model_name = "alex-test-model"
    test_model_version = "v0-Mon Apr 10 12:30:55 CST 2023"
    total_latency = 1.0
    avg_latency = 1.0
    total_request_num = 1
    current_qps = 1
    avg_qps = 1
    timestamp = time.time()
    device_id = 100
    FedMLModelDatabase.get_instance().set_monitor_metrics(test_end_point_id,
                                                          test_end_point_name,
                                                          test_model_name,
                                                          test_model_version,
                                                          total_latency, avg_latency,
                                                          total_request_num, current_qps,
                                                          avg_qps, timestamp, device_id)

    ret_latest_metrics = FedMLModelDatabase.get_instance().get_latest_monitor_metrics(test_end_point_name,
                                                                                      test_model_name,
                                                                                      test_model_version)
    if ret_latest_metrics is None:
        print("Failed to test for setting and getting end point monitoring metrics")
    else:
        metrics_obj = json.loads(ret_latest_metrics)
        if metrics_obj["total_latency"] == total_latency and metrics_obj["avg_latency"] == avg_latency and \
                metrics_obj["total_request_num"] == total_request_num and metrics_obj["current_qps"] == current_qps and \
                metrics_obj["avg_qps"] == avg_qps and metrics_obj["timestamp"] == timestamp:
            print("Succeeded to test for setting and getting end point monitoring metrics")
        else:
            print("Failed to test for setting and getting end point monitoring metrics")

    ret_latest_metrics = FedMLModelDatabase.get_instance().get_monitor_metrics_item(test_end_point_name,
                                                                                    test_model_name,
                                                                                    test_model_version, 0)
    if ret_latest_metrics is None:
        print("Failed to test for setting and getting end point monitoring metrics")
    else:
        metrics_obj = json.loads(ret_latest_metrics)
        if metrics_obj["total_latency"] == total_latency and metrics_obj["avg_latency"] == avg_latency and \
                metrics_obj["total_request_num"] == total_request_num and metrics_obj["current_qps"] == current_qps and \
                metrics_obj["avg_qps"] == avg_qps and metrics_obj["timestamp"] == timestamp:
            print("Succeeded to test for setting and getting end point monitoring metrics")
        else:
            print("Failed to test for setting and getting end point monitoring metrics")


if __name__ == "__main__":
    test_deployment_result()

    test_deployment_status()

    test_end_point_status()

    test_end_point_activation()

    test_end_point_device_info()

    test_end_point_token()

    test_end_point_monitor_metrics()

    pass
