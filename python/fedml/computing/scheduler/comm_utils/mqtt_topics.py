from typing import Union


class MqttTopics:
    # The MQTT message topic format is as follows: <sender>/<receiver>/<action>

    __last_will_message = "flclient_agent/last_will_msg"

    # ============== Server -> Client ==============

    # Train Topics
    __server_client_start_train = "flserver_agent/{client_id}/start_train"
    __server_client_stop_train = "flserver_agent/{client_id}/stop_train"

    # Device Monitoring Topics
    __server_client_request_device_info = "server/client/request_device_info/{client_id}"
    __client_client_agent_status = "fl_client/flclient_agent_{client_id}/status"
    __server_server_agent_status = "fl_server/flserver_agent_{server_id}/status"

    # ============== Client -> Server ==============

    # Metrics and Logs Topics
    __client_server_metrics = "fedml_slave/fedml_master/metrics/{run_id}"
    __client_server_logs = "fedml_slave/fedml_master/logs/{run_id}"
    __client_server_response_device_info = "client/server/response_device_info/{server_id}"

    # ============== MLOps -> Client ==============

    # Authentication Topics
    __mlops_client_logout = "mlops/client/logout/{client_id}"

    # Device Monitoring and Library Update Topics
    __mlops_client_report_device_status = "mlops/report_device_status"
    __mlops_client_ota = "mlops/flclient_agent_{client_id}/ota"

    # Deployment Topics
    __deploy_mlops_slave_request_device_info = "deploy/mlops/slave_agent/request_device_info/{slave_id}"
    __deploy_mlops_client_request_device_info = "deploy/mlops/client_agent/request_device_info/{client_id}"

    # ============== MLOps -> Server ==============
    __mlops_server_start_train = "mlops/flserver_agent_{server_id}/start_train"
    __mlops_server_stop_train = "mlops/flserver_agent_{server_id}/stop_train"
    __mlops_server_ota = "mlops/flserver_agent_{server_id}/ota"

    # Deployment Topics
    __mlops_master_request_device_info = "deploy/mlops/master_agent/request_device_info/{master_id}"
    __deploy_mlops_master_request_device_info = "deploy/mlops/master_agent/request_device_info/{master_id}"

    # ============== Client -> MLOps ==============

    # Monitoring Topics
    __client_mlops_status = "fl_client/mlops/status"
    __client_mlops_active = "flclient_agent/active"

    # Run Topics
    __run_client_mlops_status = "fl_run/fl_client/mlops/status"
    __run_server_mlops_status = "fl_run/fl_server/mlops/status"
    __client_mlops_job_cost = "ml_client/mlops/job_computing_cost"

    # ============== Server -> MLOps ==============

    # Train Topics
    __server_mlops_training_progress = "fl_server/mlops/training_progress_and_eval"
    # TODO (alaydshah): Fix the typo (roundx -> rounds)
    __server_mlops_training_rounds = "fl_server/mlops/training_roundx"

    # Federate Topics
    __server_mlops_client_model = "fl_server/mlops/client_model"
    __server_mlops_aggregated_model = "fl_server/mlops/global_aggregated_model"
    __server_mlops_training_model_net = "fl_server/mlops/training_model_net"

    # Deploy Topics
    __server_mlops_deploy_progress = "fl_server/mlops/deploy_progress_and_eval"
    __model_serving_mlops_llm_input_output_record = "model_serving/mlops/llm_input_output_record"
    __deploy_master_mlops_response_device_info = "deploy/master_agent/mlops/response_device_info"

    # Monitoring Topics
    __server_mlops_active = "flserver_agent/active"

    # ============== Device -> MLOps ==============

    # Launch Topics
    __launch_mlops_artifacts = "launch_device/mlops/artifacts"
    __launch_mlops_release_gpu_ids = "launch_device/mlops/release_gpu_ids"
    __launch_mlops_sync_deploy_ids = "launch_device/mlops/sync_deploy_ids"

    # Deployment Topics
    __deploy_mlops_status = "model_ops/model_device/return_deployment_status"
    __compute_mlops_endpoint = "compute/mlops/endpoint"

    # Device & System Performance Topics
    __client_mlops_system_performance = "fl_client/mlops/system_performance"
    __client_mlops_gpu_device_info = "ml_client/mlops/gpu_device_info"

    # ============== Diagnosis ==============
    __test_mqtt_connection = "fedml/{mqtt_client_id}/test_mqtt_msg"

    # TODO (alaydshah): Make sure these aren't used anywhere, and clean them up
    # ============== Deprecated ==============

    __server_run_exception = "flserver_agent/{run_id}/client_exit_train_with_exception"
    __server_mlops_status = "fl_server/mlops/status"
    __client_mlops_training_metrics = "fl_client/mlops/training_metrics"
    __mlops_runtime_logs_run = "mlops/runtime_logs/{run_id}"

    @classmethod
    def server_client_start_train(cls, client_id: Union[int, str]):
        return cls.__server_client_start_train.format(client_id=client_id)

    @classmethod
    def server_client_stop_train(cls, client_id: Union[int, str]):
        return cls.__server_client_stop_train.format(client_id=client_id)

    @classmethod
    def server_client_request_device_info(cls, client_id: Union[int, str]):
        return cls.__server_client_request_device_info.format(client_id=client_id)

    @classmethod
    def client_client_agent_status(cls, client_id: Union[int, str]):
        return cls.__client_client_agent_status.format(client_id=client_id)

    @classmethod
    def server_server_agent_status(cls, server_id: Union[int, str]):
        return cls.__server_server_agent_status.format(server_id=server_id)

    @classmethod
    def mlops_client_report_device_status(cls):
        return cls.__mlops_client_report_device_status

    @classmethod
    def mlops_client_ota(cls, client_id: Union[int, str]):
        return cls.__mlops_client_ota.format(client_id=client_id)

    @classmethod
    def deploy_mlops_slave_request_device_info(cls, slave_id: Union[int, str]):
        return cls.__deploy_mlops_slave_request_device_info.format(slave_id=slave_id)

    @classmethod
    def mlops_master_request_device_info(cls, master_id: Union[int, str]):
        return cls.__mlops_master_request_device_info.format(master_id=master_id)

    @classmethod
    def mlops_client_logout(cls, client_id: Union[int, str]):
        return cls.__mlops_client_logout.format(client_id=client_id)

    @classmethod
    def deploy_mlops_client_request_device_info(cls, client_id: Union[int, str]):
        return cls.__deploy_mlops_client_request_device_info.format(client_id=client_id)

    @classmethod
    def last_will_message(cls):
        return cls.__last_will_message

    @classmethod
    def client_mlops_status(cls):
        return cls.__client_mlops_status

    @classmethod
    def run_client_mlops_status(cls):
        return cls.__run_client_mlops_status

    @classmethod
    def run_server_mlops_status(cls):
        return cls.__run_server_mlops_status

    @classmethod
    def server_run_exception(cls, run_id: Union[int, str]):
        return cls.__server_run_exception.format(run_id=run_id)

    @classmethod
    def server_mlops_status(cls):
        return cls.__server_mlops_status

    @classmethod
    def client_mlops_training_metrics(cls):
        return cls.__client_mlops_training_metrics

    @classmethod
    def server_mlops_training_progress(cls):
        return cls.__server_mlops_training_progress

    @classmethod
    def server_mlops_deploy_progress(cls):
        return cls.__server_mlops_deploy_progress

    @classmethod
    def mlops_server_ota(cls, server_id: Union[int, str]):
        return cls.__mlops_server_ota.format(server_id=server_id)

    @classmethod
    def client_server_metrics(cls, run_id: Union[int, str]):
        return cls.__client_server_metrics.format(run_id=run_id)

    @classmethod
    def client_server_logs(cls, run_id: Union[int, str]):
        return cls.__client_server_logs.format(run_id=run_id)

    @classmethod
    def server_mlops_training_rounds(cls):
        return cls.__server_mlops_training_rounds

    @classmethod
    def server_mlops_client_model(cls):
        return cls.__server_mlops_client_model

    @classmethod
    def server_mlops_aggregated_model(cls):
        return cls.__server_mlops_aggregated_model

    @classmethod
    def server_mlops_training_model_net(cls):
        return cls.__server_mlops_training_model_net

    @classmethod
    def model_serving_mlops_llm_input_output_record(cls):
        return cls.__model_serving_mlops_llm_input_output_record

    @classmethod
    def client_mlops_job_cost(cls):
        return cls.__client_mlops_job_cost

    @classmethod
    def mlops_runtime_logs_run(cls, run_id: Union[int, str]):
        return cls.__mlops_runtime_logs_run.format(run_id=run_id)

    @classmethod
    def launch_mlops_artifacts(cls):
        return cls.__launch_mlops_artifacts

    @classmethod
    def deploy_mlops_status(cls):
        return cls.__deploy_mlops_status

    @classmethod
    def client_mlops_system_performance(cls):
        return cls.__client_mlops_system_performance

    @classmethod
    def client_mlops_gpu_device_info(cls):
        return cls.__client_mlops_gpu_device_info

    @classmethod
    def compute_mlops_endpoint(cls):
        return cls.__compute_mlops_endpoint

    @classmethod
    def launch_mlops_release_gpu_ids(cls):
        return cls.__launch_mlops_release_gpu_ids

    @classmethod
    def launch_mlops_sync_deploy_ids(cls):
        return cls.__launch_mlops_sync_deploy_ids

    @classmethod
    def mlops_server_start_train(cls, server_id: Union[int, str]):
        return cls.__mlops_server_start_train.format(server_id=server_id)

    @classmethod
    def mlops_server_stop_train(cls, server_id: Union[int, str]):
        return cls.__mlops_server_stop_train.format(server_id=server_id)

    @classmethod
    def client_server_response_device_info(cls, server_id: Union[int, str]):
        return cls.__client_server_response_device_info.format(server_id=server_id)

    @classmethod
    def deploy_mlops_master_request_device_info(cls, master_id: Union[int, str]):
        return cls.__deploy_mlops_master_request_device_info.format(master_id=master_id)

    @classmethod
    def client_mlops_active(cls):
        return cls.__client_mlops_active

    @classmethod
    def server_mlops_active(cls):
        return cls.__server_mlops_active

    @classmethod
    def deploy_master_mlops_response_device_info(cls):
        return cls.__deploy_master_mlops_response_device_info

    @classmethod
    def test_mqtt_connection(cls, mqtt_client_id: Union[int, str]):
        return cls.__test_mqtt_connection.format(mqtt_client_id=mqtt_client_id)
