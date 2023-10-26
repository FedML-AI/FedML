
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
