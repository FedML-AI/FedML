import os

import fedml


class SchedulerConstants:
    ERR_MSG_BINDING_EXCEPTION_1 = "[1] Exception occurs when logging to MLOps."
    ERR_MSG_BINDING_EXCEPTION_2 = "[2] Exception occurs when logging to MLOps."
    ERR_MSG_BINDING_EXIT_RETRYING = "If you don't want to retry logging into MLOps, open another terminal and "\
                                    "run `fedml logout` to logout."

    PLATFORM_TYPE_FALCON = "falcon"
    PLATFORM_TYPE_OCTOPUS = "octopus"

    JOB_PACKAGE_TYPE_TRAIN = "train/build"
    JOB_PACKAGE_TYPE_FEDERATE = "federate/build"
    JOB_PACKAGE_TYPE_LAUNCH = "launch/build"
    JOB_PACKAGE_TYPE_DEFAULT = "default"

    LAUNCH_JOB_DEFAULT_ENTRY_NAME = "fedml_job_entry_pack.sh"
    LAUNCH_SERVER_JOB_DEFAULT_ENTRY_NAME = "fedml_server_job_entry_pack.sh"

    CLIENT_SHELL_BASH = "bash"
    CLIENT_SHELL_PS = "powershell"

    JOB_MATCH_DEFAULT_MASTER_NODE_PORT = 40000

    JOB_TASK_TYPE_TRAIN = "train"
    JOB_TASK_TYPE_DEPLOY = "deploy"
    JOB_TASK_TYPE_SERVE = "serve"
    JOB_TASK_TYPE_FEDERATE = "federate"
    JOB_TASK_TYPE_DEV_ENV = "dev-environment"

    JOB_YAML_RESERVED_CONFIG_KEY_WORDS = [
        "workspace", "job", "computing", "fedml_env", "bootstrap", "job_type", "job_subtype",
        "framework_type", "server_job", "job_args", "job_name", "serving_args"]

    MLOPS_CLIENT_STATUS_OFFLINE = "OFFLINE"
    MLOPS_CLIENT_STATUS_IDLE = "IDLE"
    MLOPS_CLIENT_STATUS_UPGRADING = "UPGRADING"
    MLOPS_CLIENT_STATUS_QUEUED = "QUEUED"
    MLOPS_CLIENT_STATUS_INITIALIZING = "INITIALIZING"
    MLOPS_CLIENT_STATUS_TRAINING = "TRAINING"
    MLOPS_CLIENT_STATUS_STOPPING = "STOPPING"
    MLOPS_CLIENT_STATUS_KILLED = "KILLED"
    MLOPS_CLIENT_STATUS_FAILED = "FAILED"
    MLOPS_CLIENT_STATUS_FINISHED = "FINISHED"

    MLOPS_SERVER_DEVICE_STATUS_OFFLINE = "OFFLINE"
    MLOPS_SERVER_DEVICE_STATUS_IDLE = "IDLE"
    MLOPS_SERVER_DEVICE_STATUS_STARTING = "STARTING"
    MLOPS_SERVER_DEVICE_STATUS_RUNNING = "RUNNING"
    MLOPS_SERVER_DEVICE_STATUS_STOPPING = "STOPPING"
    MLOPS_SERVER_DEVICE_STATUS_KILLED = "KILLED"
    MLOPS_SERVER_DEVICE_STATUS_FAILED = "FAILED"
    MLOPS_SERVER_DEVICE_STATUS_FINISHED = "FINISHED"

    # Device Status
    MLOPS_DEVICE_STATUS_IDLE = "IDLE"
    MLOPS_DEVICE_STATUS_UPGRADING = "UPGRADING"
    MLOPS_DEVICE_STATUS_RUNNING = "RUNNING"
    MLOPS_DEVICE_STATUS_OFFLINE = "OFFLINE"

    # Run Status
    MLOPS_RUN_STATUS_QUEUED = "QUEUED"
    MLOPS_RUN_STATUS_STARTING = "STARTING"
    MLOPS_RUN_STATUS_RUNNING = "RUNNING"
    MLOPS_RUN_STATUS_STOPPING = "STOPPING"
    MLOPS_RUN_STATUS_KILLED = "KILLED"
    MLOPS_RUN_STATUS_FAILED = "FAILED"
    MLOPS_RUN_STATUS_FINISHED = "FINISHED"
    MLOPS_RUN_STATUS_IDLE = "IDLE"
    MLOPS_RUN_STATUS_OFFLINE = "OFFLINE"

    ENDPOINT_FAIL_THRESHOLD_VALUE = 3
    ENDPOINT_DEPLOYMENT_PROVISIONING_TIMEOUT = 60 * 25
    ENDPOINT_DEPLOYMENT_DEPLOYING_TIMEOUT = 60 * 60 * 2
    ENDPOINT_INFERENCE_READY_TIMEOUT = 15
    ENDPOINT_STATUS_CHECK_TIMEOUT = 60 * 3

    MQTT_INFERENCE_TIMEOUT = 60 * 6

    TRAIN_PROVISIONING_TIMEOUT = 60 * 25
    TRAIN_STARTING_TIMEOUT = 60 * 15
    TRAIN_STOPPING_TIMEOUT = 60 * 5
    TRAIN_RUNNING_TIMEOUT = 60 * 60 * 12
    TRAIN_INIT_TIMEOUT = 60 * 5

    PUBLIC_REDIS_PORT = 6379
    PUBLIC_REDIS_PASSWORD = "share-fedml-secret@@DD#D#*&^"
    REDIS_CONN_TIMEOUT = 15

    STATUS_CHECK_FRO_RUN_STOP_CONTEXT = "run_stop"
    BINDING_ACCOUNT_NOT_EXIST_ERROR = "DATA_NO_EXIST_ERROR"

    MLOPS_RUN_COMPLETED_STATUS_LIST = [
        MLOPS_RUN_STATUS_FINISHED, MLOPS_RUN_STATUS_KILLED, MLOPS_RUN_STATUS_FAILED,
        MLOPS_RUN_STATUS_IDLE, MLOPS_RUN_STATUS_OFFLINE
    ]

    RUN_PROCESS_TYPE_USER_PROCESS = "user-process"
    RUN_PROCESS_TYPE_RUNNER_PROCESS = "runner-process"
    RUN_PROCESS_TYPE_BOOTSTRAP_PROCESS = "bootstrap-process"

    FEDML_DEFAULT_LAUNCH_CONTAINER_PREFIX = "fedml_default_launch_container"
    FEDML_DEFAULT_LAUNCH_IMAGE = "fedml/fedml-default-launch:cu12.1-u22.04"
    FEDML_DEFAULT_LOG_DIR = ".fedml/fedml-client/fedml/logs"
    FEDML_DEFAULT_DATA_DIR = ".fedml/fedml-client/fedml/data"

    IMAGE_PULL_POLICY_ALWAYS = "Always"
    IMAGE_PULL_POLICY_IF_NOT_PRESENT = "IfNotPresent"
    IMAGE_PULL_POLICY_NEVER = "Never"

    @staticmethod
    def get_log_source(run_json):
        run_config = run_json.get("run_config", {})
        run_params = run_config.get("parameters", {})
        job_yaml = run_params.get("job_yaml", None)
        if job_yaml is None:
            log_source_type = SchedulerConstants.JOB_TASK_TYPE_FEDERATE
        elif isinstance(job_yaml, dict):
            job_type = job_yaml.get("job_type", None)
            job_type = job_yaml.get("task_type", SchedulerConstants.JOB_TASK_TYPE_TRAIN) \
                if job_type is None else job_type
            if str(job_type).strip() == "":
                log_source_type = SchedulerConstants.JOB_TASK_TYPE_TRAIN
            else:
                log_source_type = job_type
                if job_type == SchedulerConstants.JOB_TASK_TYPE_SERVE:
                    log_source_type = SchedulerConstants.JOB_TASK_TYPE_DEPLOY
        else:
            log_source_type = SchedulerConstants.JOB_TASK_TYPE_FEDERATE

        return log_source_type

    @staticmethod
    def is_run_completed(status):
        return True if status in SchedulerConstants.MLOPS_RUN_COMPLETED_STATUS_LIST else False

    @staticmethod
    def get_redis_and_infer_host_env_addr():
        infer_host = os.getenv("FEDML_INFER_HOST", None)
        infer_redis_addr = os.getenv("FEDML_INFER_REDIS_ADDR", None)
        infer_redis_port = os.getenv("FEDML_INFER_REDIS_PORT", None)
        infer_redis_password = os.getenv("FEDML_INFER_REDIS_PASSWORD", None)
        disable_redis = os.getenv("FEDML_DISABLE_REDIS_CONNECTION", None)
        return infer_host, infer_redis_addr, infer_redis_port, infer_redis_password, disable_redis

    @staticmethod
    def get_public_redis_addr():
        return "cache{}.fedml.ai".format(
            "-" + fedml.get_env_version() if fedml.get_env_version() != "release" else "")

    @staticmethod
    def is_deploy_job(job_type):
        return True if job_type is not None and job_type == SchedulerConstants.JOB_TASK_TYPE_DEPLOY else False
