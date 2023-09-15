import os

import click

from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.computing.scheduler.slave.client_login import logout as client_logout
from .utils import stop_jobs_core


@click.group("launch", help="Launch job at the FedML® Launch platform (open.fedml.ai)",
             context_settings={"ignore_unknown_options": True}, invoke_without_command=True)
@click.help_option("--help", "-h")
@click.argument("yaml_file", nargs=-1)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--group",
    "-g",
    type=str,
    default="",
    help="The queue group id on which your job will be scheduled.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="launch job to which version of MLOps platform. It should be dev, test or release",
)
@click.option(
    "--cluster",
    "-c",
    type=str,
    help="Please provide a cluster name. If a cluster with that name already exists, it will be used; otherwise, "
         "a new cluster with the provided name will be created."
)
def launch(yaml_file, api_key, group, cluster, version):
    """
    Manage resources on the FedML® Launch platform (open.fedml.ai).
    """

    if not yaml_file and click.get_current_context().invoked_subcommand is None:
        # No subcommand and no yaml_file provided, display help
        click.echo("No subcommand or yaml_file provided. Displaying help...")
        ctx = click.get_current_context()
        click.echo(ctx.get_help())

    else:
        error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
        if error_code != 0:
            click.echo("Please check if your API key is valid.")
            return

        FedMLLaunchManager.get_instance().set_config_version(version)
        FedMLLaunchManager.get_instance().api_launch_job(yaml_file[0], None)


@launch.command("login", help="Login to the FedML® Launch platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.argument("userid", nargs=-1)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="login to which version of FedML® Launch platform. It should be dev, test or release",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
def launch_login(userid, version, api_key):
    # Check api key
    if api_key is None or api_key == "":
        saved_api_key = FedMLLaunchManager.get_api_key()
        if saved_api_key is None or saved_api_key == "":
            api_key = click.prompt("FedML® Launch API Key is not set yet, please input your API key", hide_input=True)
        else:
            api_key = saved_api_key

    print("\n Welcome to FedML.ai! \n Start to login the current device to the MLOps (https://open.fedml.ai)...\n")
    if userid is None or len(userid) <= 0:
        click.echo("Please specify your account id, usage: fedml launch login $your_account_id -k $your_api_key")
        return
    account_id = userid[0]
    platform_url = "open.fedml.ai"
    if version != "release":
        platform_url = "open-{}.fedml.ai".format(version)

    # Check user id.
    if userid == "":
        click.echo(
            "Please provide your account id in the MLOps platform ({}).".format(
                platform_url
            )
        )
        return

    pip_source_dir = os.path.dirname(__file__)
    pip_source_dir = os.path.dirname(pip_source_dir)
    login_cmd = os.path.join(pip_source_dir, "computing", "scheduler", "slave", "client_daemon.py")

    client_logout()
    sys_utils.cleanup_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                    ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
    sys_utils.cleanup_all_fedml_client_learning_processes()
    sys_utils.cleanup_all_fedml_client_login_processes("client_login.py")
    sys_utils.cleanup_all_fedml_client_api_processes(kill_all=True)
    role = ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_GPU_SUPPLIER_INDEX]

    login_pid = sys_utils.run_subprocess_open(
        [
            sys_utils.get_python_program(),
            "-W",
            "ignore",
            login_cmd,
            "-t",
            "login",
            "-u",
            str(account_id),
            "-v",
            version,
            "-ls",
            "127.0.0.1",
            "-r",
            role,
            "-id",
            "0",
            "-os",
            "",
            "-k",
            api_key,
            "-ngc",
            "1"
        ]
    ).pid
    sys_utils.save_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                 ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME, login_pid)


@launch.command("logout", help="Logout from the FedML® Launch platform (open.fedml.ai)")
@click.help_option("--help", "-h")
def launch_logout():
    sys_utils.cleanup_all_fedml_client_login_processes("client_daemon.py")
    client_logout()
    sys_utils.cleanup_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                    ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
    sys_utils.cleanup_all_fedml_client_learning_processes()
    sys_utils.cleanup_all_fedml_client_login_processes("client_login.py")
    sys_utils.cleanup_all_fedml_client_api_processes(kill_all=True)
    sys_utils.cleanup_all_fedml_client_login_processes("client_daemon.py")

    print("\nlogout successfully!\n")


@launch.command("cancel", help="Cancel job at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("job_id", nargs=-1)
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="stop a job at which version of FedML® Launch platform. It should be dev, test or release",
)
def launch_cancel(job_id, platform, api_key, version):
    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    api_key = FedMLLaunchManager.get_api_key()
    stop_jobs_core(platform, job_id[0], api_key, version)


@launch.command("log", help="View the job list at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("job_id", nargs=-1)
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="list jobs at which version of the FedML® Launch platform. It should be dev, test or release",
)
def launch_log(job_id, platform, api_key, version):
    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    FedMLLaunchManager.get_instance().api_launch_log(job_id[0], 0, 0, need_all_logs=True)


@launch.command("queue", help="View the job queue at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("group_id", nargs=-1)
def launch_queue(group_id):
    pass
