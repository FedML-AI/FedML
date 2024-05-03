from fedml.computing.scheduler.comm_utils.mqtt_topics import MqttTopics


def test_mqtt_topics():
    mqtt_client_id = 0
    edge_id = 1
    server_id = 2
    end_point_id = 10
    run_id = 100
    model_device_client_edge_id_list = [1, 2, 3]

    topic_start_train = "flserver_agent/" + str(edge_id) + "/start_train"
    assert MqttTopics.server_client_start_train(client_id=edge_id) == topic_start_train

    topic_stop_train = "flserver_agent/" + str(edge_id) + "/stop_train"
    assert MqttTopics.server_client_stop_train(client_id=edge_id) == topic_stop_train

    topic_client_status = "fl_client/flclient_agent_" + str(edge_id) + "/status"
    assert MqttTopics.client_client_agent_status(client_id=edge_id) == topic_client_status

    topic_report_status = "mlops/report_device_status"
    assert MqttTopics.mlops_client_report_device_status() == topic_report_status

    topic_ota_msg = "mlops/flclient_agent_" + str(edge_id) + "/ota"
    assert MqttTopics.mlops_client_ota(client_id=edge_id) == topic_ota_msg

    topic_request_device_info = "server/client/request_device_info/" + str(edge_id)
    assert MqttTopics.server_client_request_device_info(client_id=edge_id) == topic_request_device_info

    topic_request_edge_device_info_from_mlops = f"deploy/mlops/slave_agent/request_device_info/{edge_id}"
    assert MqttTopics.deploy_mlops_slave_request_device_info(
        slave_id=edge_id) == topic_request_edge_device_info_from_mlops

    topic_request_client_device_info_from_mlops = f"deploy/mlops/client_agent/request_device_info/{edge_id}"
    assert (MqttTopics.deploy_mlops_client_request_device_info(client_id=edge_id)
            == topic_request_client_device_info_from_mlops)

    topic_deploy_master_request_device_info = f"deploy/mlops/master_agent/request_device_info/{edge_id}"
    assert (MqttTopics.deploy_mlops_master_request_device_info(master_id=edge_id) ==
            topic_deploy_master_request_device_info)

    topic_request_deploy_slave_device_info_from_mlops = (f"deploy/mlops/slave_agent/request_device_info/"
                                                         f"{model_device_client_edge_id_list[0]}")
    assert (MqttTopics.deploy_mlops_slave_request_device_info
            (slave_id=model_device_client_edge_id_list[0]) == topic_request_deploy_slave_device_info_from_mlops)

    topic_client_logout = "mlops/client/logout/" + str(edge_id)
    assert MqttTopics.mlops_client_logout(client_id=edge_id) == topic_client_logout

    topic_last_will_message = "flclient_agent/last_will_msg"
    assert MqttTopics.last_will_message() == topic_last_will_message

    topic_client_mlops_statis = "fl_client/mlops/status"
    assert MqttTopics.client_mlops_status() == topic_client_mlops_statis

    topic_run_client_mlops_status = "fl_run/fl_client/mlops/status"
    assert MqttTopics.run_client_mlops_status() == topic_run_client_mlops_status

    topic_run_server_mlops_status = "fl_run/fl_server/mlops/status"
    assert MqttTopics.run_server_mlops_status() == topic_run_server_mlops_status

    topic_server_server_agent_status = f"fl_server/flserver_agent_{server_id}/status"
    assert MqttTopics.server_server_agent_status(server_id=server_id) == topic_server_server_agent_status

    topic_exit_train_with_exception = "flserver_agent/" + str(run_id) + "/client_exit_train_with_exception"
    assert MqttTopics.server_run_exception(run_id=run_id) == topic_exit_train_with_exception

    topic_server_mlops_status = "fl_server/mlops/status"
    assert MqttTopics.server_mlops_status() == topic_server_mlops_status

    topic_client_mlops_training_metrics = "fl_client/mlops/training_metrics"
    assert MqttTopics.client_mlops_training_metrics() == topic_client_mlops_training_metrics

    topic_server_mlops_training_progress = "fl_server/mlops/training_progress_and_eval"
    assert MqttTopics.server_mlops_training_progress() == topic_server_mlops_training_progress

    topic_server_mlops_deploy_progress = "fl_server/mlops/deploy_progress_and_eval"
    assert MqttTopics.server_mlops_deploy_progress() == topic_server_mlops_deploy_progress

    topic_client_server_metrics = f"fedml_slave/fedml_master/metrics/{run_id}"
    assert MqttTopics.client_server_metrics(run_id=run_id) == topic_client_server_metrics

    topic_client_server_logs = f"fedml_slave/fedml_master/logs/{run_id}"
    assert MqttTopics.client_server_logs(run_id=run_id) == topic_client_server_logs

    topic_server_mlops_training_rounds = "fl_server/mlops/training_roundx"
    assert MqttTopics.server_mlops_training_rounds() == topic_server_mlops_training_rounds

    topic_server_mlops_client_model = "fl_server/mlops/client_model"
    assert MqttTopics.server_mlops_client_model() == topic_server_mlops_client_model

    topic_server_mlops_aggregated_model = "fl_server/mlops/global_aggregated_model"
    assert MqttTopics.server_mlops_aggregated_model() == topic_server_mlops_aggregated_model

    topic_server_mlops_training_model_net = "fl_server/mlops/training_model_net"
    assert MqttTopics.server_mlops_training_model_net() == topic_server_mlops_training_model_net

    topic_model_serving_mlops_llm_input_output = "model_serving/mlops/llm_input_output_record"
    assert MqttTopics.model_serving_mlops_llm_input_output_record() == topic_model_serving_mlops_llm_input_output

    topic_client_mlops_job_cost = "ml_client/mlops/job_computing_cost"
    assert MqttTopics.client_mlops_job_cost() == topic_client_mlops_job_cost

    topic_mlops_runtime_logs_run = "mlops/runtime_logs/" + str(run_id)
    assert MqttTopics.mlops_runtime_logs_run(run_id=run_id) == topic_mlops_runtime_logs_run

    topic_launch_mlops_artifacts = "launch_device/mlops/artifacts"
    assert MqttTopics.launch_mlops_artifacts() == topic_launch_mlops_artifacts

    deployment_status_topic_prefix = "model_ops/model_device/return_deployment_status"
    assert MqttTopics.deploy_mlops_status() == deployment_status_topic_prefix

    topic_client_mlops_system_performance = "fl_client/mlops/system_performance"
    assert MqttTopics.client_mlops_system_performance() == topic_client_mlops_system_performance

    topic_client_mlops_gpu_device_info = "ml_client/mlops/gpu_device_info"
    assert MqttTopics.client_mlops_gpu_device_info() == topic_client_mlops_gpu_device_info

    topic_compute_mlops_endpoint = "compute/mlops/endpoint"
    assert MqttTopics.compute_mlops_endpoint() == topic_compute_mlops_endpoint

    topic_launch_mlops_release_gpu_ids = "launch_device/mlops/release_gpu_ids"
    assert MqttTopics.launch_mlops_release_gpu_ids() == topic_launch_mlops_release_gpu_ids

    topic_launch_mlops_sync_deploy_ids = "launch_device/mlops/sync_deploy_ids"
    assert MqttTopics.launch_mlops_sync_deploy_ids() == topic_launch_mlops_sync_deploy_ids

    topic_server_start_train = "mlops/flserver_agent_" + str(server_id) + "/start_train"
    assert MqttTopics.mlops_server_start_train(server_id=server_id) == topic_server_start_train

    topic_stop_train = "mlops/flserver_agent_" + str(server_id) + "/stop_train"
    assert MqttTopics.mlops_server_stop_train(server_id=server_id) == topic_stop_train

    topic_server_ota = "mlops/flserver_agent_" + str(server_id) + "/ota"
    assert MqttTopics.mlops_server_ota(server_id=server_id) == topic_server_ota

    topic_response_device_info = "client/server/response_device_info/" + str(server_id)
    assert MqttTopics.client_server_response_device_info(server_id=server_id) == topic_response_device_info

    topic_agent_mlops_active = "flclient_agent/active"
    assert MqttTopics.client_mlops_active() == topic_agent_mlops_active

    topic_server_mlops_active = "flserver_agent/active"
    assert MqttTopics.server_mlops_active() == topic_server_mlops_active

    topic_master_mlops_response_device_info = "deploy/master_agent/mlops/response_device_info"
    assert MqttTopics.deploy_master_mlops_response_device_info() == topic_master_mlops_response_device_info

    topic_test_mqtt_connection = "fedml/" + str(mqtt_client_id) + "/test_mqtt_msg"
    assert MqttTopics.test_mqtt_connection(mqtt_client_id=mqtt_client_id) == topic_test_mqtt_connection
