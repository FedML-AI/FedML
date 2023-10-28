
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

