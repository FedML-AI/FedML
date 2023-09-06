import os
import json
import shutil
from os.path import expanduser

import click

import fedml

from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.computing.scheduler.scheduler_entry.constants import Constants
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.slave.client_login import logout as client_logout
from fedml.computing.scheduler.env.collect_env import collect_env
from fedml.computing.scheduler.master.server_login import logout as server_logout
from fedml.computing.scheduler.slave.docker_login import login_with_docker_mode
from fedml.computing.scheduler.slave.docker_login import logout_with_docker_mode
from fedml.computing.scheduler.slave.docker_login import logs_with_docker_mode
from fedml.computing.scheduler.master.docker_login import login_with_server_docker_mode
from fedml.computing.scheduler.master.docker_login import logout_with_server_docker_mode
from fedml.computing.scheduler.master.docker_login import logs_with_server_docker_mode
from fedml.computing.scheduler.slave.client_diagnosis import ClientDiagnosis
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.model_scheduler import device_login_entry
from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards
from fedml.computing.scheduler.scheduler_entry.job_manager import FedMLJobManager
from fedml.computing.scheduler.comm_utils.platform_utils import platform_is_valid
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager

from prettytable import PrettyTable

FEDML_MLOPS_BUILD_PRE_IGNORE_LIST = 'dist-packages,client-package.zip,server-package.zip,__pycache__,*.pyc,*.git'
simulator_process_list = list()


@click.group()
@click.help_option("--help", "-h")
def cli():
    pass


@cli.command("version", help="Display fedml version.")
@click.help_option("--help", "-h")
def mlops_version():
    click.echo("fedml version: " + str(fedml.__version__))


@cli.command("status", help="Display fedml client training status.")
@click.help_option("--help", "-h")
def mlops_status():
    training_infos = ClientConstants.get_training_infos()
    click.echo(
        "Client training status: " + str(training_infos["training_status"]).upper()
    )


@cli.command("show-resource-type", help="Show resource type at the FedML® Launch platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="show resource type at which version of MLOps platform. It should be dev, test or release",
)
def launch_show_resource_type(version):
    FedMLLaunchManager.get_instance().set_config_version(version)
    resource_type_list = FedMLLaunchManager.get_instance().show_resource_type()
    if resource_type_list is not None and len(resource_type_list) > 0:
        click.echo("All available resource type is as follows.")
        resource_table = PrettyTable(['Resource Type Name'])
        for type_item in resource_type_list:
            resource_table.add_row([type_item])
        print(resource_table)
    else:
        click.echo("No available resource type.")


@cli.command("logs", help="Display fedml logs.")
@click.help_option("--help", "-h")
@click.option(
    "--client", "-c", default=None, is_flag=True, help="Display client logs.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="Display server logs.",
)
@click.option(
    "--docker", "-d", default=None, is_flag=True, help="Display client docker logs.",
)
@click.option(
    "--docker-rank", "-dr", default="1", help="docker client rank index (from 1 to n).",
)
def mlops_logs(client, server, docker, docker_rank):
    is_client = client
    is_server = server
    if client is None and server is None:
        is_client = True

    is_docker = docker
    if docker is None:
        is_docker = False

    if is_client:
        if is_docker:
            logs_with_docker_mode(docker_rank)
            return
        display_client_logs()

    if is_server:
        if is_docker:
            logs_with_server_docker_mode(docker_rank)
            return
        display_server_logs()


def display_client_logs():
    run_id, edge_id = sys_utils.get_running_info(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                                 ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
    home_dir = expanduser("~")
    log_file = "{}/{}/fedml/logs/fedml-run-{}-edge-{}.log".format(
        home_dir, ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME, str(run_id), str(edge_id)
    )

    if os.path.exists(log_file):
        with open(log_file) as file_handle:
            log_lines = file_handle.readlines()
        for log_line in log_lines:
            click.echo(log_line, nl=False)
    print("\nconsole log file path = {}".format(log_file))


def display_server_logs():
    run_id, edge_id = sys_utils.get_running_info(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                                 ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)
    home_dir = expanduser("~")
    log_file = "{}/{}/fedml/logs/fedml-run-{}-edge-{}.log".format(
        home_dir, ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME, str(run_id), str(edge_id)
    )
    if os.path.exists(log_file):
        with open(log_file) as file_handle:
            log_lines = file_handle.readlines()
        for log_line in log_lines:
            click.echo(log_line)
    print("\nconsole log file path = {}".format(log_file))


@cli.command("login", help="Login to MLOps platform")
@click.help_option("--help", "-h")
@click.argument("userid", nargs=-1)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="login to which version of MLOps platform. It should be dev, test or release",
)
@click.option(
    "--client", "-c", default=None, is_flag=True, help="login as the FedML client.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="login as the FedML server.",
)
@click.option(
    "--api_key", "-k", type=str, default="", help="user api key.",
)
@click.option(
    "--local_server",
    "-ls",
    type=str,
    default="127.0.0.1",
    help="local server address.",
)
@click.option(
    "--role",
    "-r",
    type=str,
    default="edge_server",
    help="run as the role (options: edge_server, cloud_agent, cloud_server, edge_simulator, gpu_master_server.",
)
@click.option(
    "--runner_cmd",
    "-rc",
    type=str,
    default="{}",
    help="runner commands (options: request json for start run, stop run).",
)
@click.option(
    "--device_id", "-id", type=str, default="0", help="device id.",
)
@click.option(
    "--os_name", "-os", type=str, default="", help="os name.",
)
@click.option(
    "--docker", "-d", default=None, is_flag=True, help="login with docker mode at the client agent.",
)
@click.option(
    "--docker-rank", "-dr", default="1", help="docker client rank index (from 1 to n).",
)
def mlops_login(
        userid, version, client, server,
        api_key, local_server, role, runner_cmd, device_id, os_name,
        docker, docker_rank
):
    print("\n Welcome to FedML.ai! \n Start to login the current device to the MLOps (https://open.fedml.ai)...\n")
    if userid is None or len(userid) <= 0:
        click.echo("Please specify your account id, usage: fedml login $your_account_id")
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
    # click.echo("client {}, server {}".format(client, server))
    # Set client as default entity.
    is_client = client
    is_server = server
    if client is None and server is None:
        is_client = True

    # Check api key
    user_api_key = api_key
    if api_key is None:
        user_api_key = "NONE"

    # Check docker mode.
    is_docker = docker
    if docker is None:
        is_docker = False

    if is_client is True:
        if is_docker:
            login_with_docker_mode(account_id, version, docker_rank)
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
        try:
            ClientConstants.login_role_list.index(role)
        except ValueError as e:
            role = ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_CLIEN_INDEX]

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
                local_server,
                "-r",
                role,
                "-id",
                device_id,
                "-os",
                os_name,
                "-k",
                user_api_key,
                "-ngc",
                "0"
            ]
        ).pid
        sys_utils.save_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                     ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME, login_pid)
    if is_server is True:
        # Check login mode.
        try:
            ServerConstants.login_role_list.index(role)
        except ValueError as e:
            click.echo(
                "Please specify login mode as follows ({}).".format(
                    str(ServerConstants.login_role_list)
                )
            )
            return

        if is_docker:
            login_with_server_docker_mode(account_id, version, docker_rank)
            return

        pip_source_dir = os.path.dirname(__file__)
        pip_source_dir = os.path.dirname(pip_source_dir)
        login_cmd = os.path.join(pip_source_dir, "computing", "scheduler", "master", "server_daemon.py")
        server_logout()
        sys_utils.cleanup_login_process(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                        ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_server_learning_processes()
        sys_utils.cleanup_all_fedml_server_login_processes("server_login.py")
        sys_utils.cleanup_all_fedml_server_api_processes(kill_all=True)
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
                local_server,
                "-r",
                role,
                "-rc",
                runner_cmd,
                "-id",
                device_id,
                "-os",
                os_name,
                "-k",
                user_api_key
            ]
        ).pid
        sys_utils.save_login_process(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                     ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME, login_pid)


class DefaultCommandGroup(click.Group):

    def __init__(self, *args, **kwargs):
        self.default_command = kwargs.pop('default_command', None)
        super().__init__(*args, **kwargs)

    def resolve_command(self, ctx, args):
        try:
            return super().resolve_command(ctx, args)
        except click.UsageError:
            args.insert(0, self.default_command)
            return super().resolve_command(ctx, args)


@cli.group("launch", cls=DefaultCommandGroup, default_command='run')
@click.help_option("--help", "-h")
def launch():
    """
    Manage resources on the FedML® Launch platform (open.fedml.ai).
    """
    pass


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


@cli.command("logout", help="Logout from MLOps platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.option(
    "--client", "-c", default=None, is_flag=True, help="logout from the FedML client.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="logout from the FedML server.",
)
@click.option(
    "--docker", "-d", default=None, is_flag=True, help="logout from docker mode at the client agent.",
)
@click.option(
    "--docker-rank", "-dr", default=None, help="docker client rank index (from 1 to n).",
)
def mlops_logout(client, server, docker, docker_rank):
    is_client = client
    is_server = server
    if client is None and server is None:
        is_client = True

    is_docker = docker
    if docker is None:
        is_docker = False

    if is_client is True:
        if is_docker:
            logout_with_docker_mode(docker_rank)
            return
        sys_utils.cleanup_all_fedml_client_login_processes("client_daemon.py")
        client_logout()
        sys_utils.cleanup_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                        ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_client_learning_processes()
        sys_utils.cleanup_all_fedml_client_login_processes("client_login.py")
        sys_utils.cleanup_all_fedml_client_api_processes(kill_all=True)
        sys_utils.cleanup_all_fedml_client_login_processes("client_daemon.py")

    if is_server is True:
        if is_docker:
            logout_with_server_docker_mode(docker_rank)
            return
        sys_utils.cleanup_all_fedml_server_login_processes("server_daemon.py")
        server_logout()
        sys_utils.cleanup_login_process(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                        ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_server_learning_processes()
        sys_utils.cleanup_all_fedml_server_login_processes("server_login.py")
        sys_utils.cleanup_all_fedml_server_api_processes(kill_all=True)
        sys_utils.cleanup_all_fedml_server_login_processes("server_daemon.py")
    print("\nlogout successfully!\n")


@cli.command("build", help="Build packages for MLOps platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="octopus",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch).",
)
@click.option(
    "--type",
    "-t",
    type=str,
    default="client",
    help="client or server? (value: client; server)",
)
@click.option(
    "--source_folder", "-sf", type=str, default="./", help="the source code folder path"
)
@click.option(
    "--entry_point",
    "-ep",
    type=str,
    default="./",
    help="the entry point of the source code",
)
@click.option(
    "--config_folder", "-cf", type=str, default="./", help="the config folder path"
)
@click.option(
    "--dest_folder",
    "-df",
    type=str,
    default="./",
    help="the destination package folder path",
)
@click.option(
    "--ignore",
    "-ig",
    type=str,
    default="",
    help="the ignore list for copying files, the format is as follows: *.model,__pycache__,*.data*, ",
)
def mlops_build(platform, type, source_folder, entry_point, config_folder, dest_folder, ignore):
    click.echo("Argument for type: " + type)
    click.echo("Argument for source folder: " + source_folder)
    click.echo("Argument for entry point: " + entry_point)
    click.echo("Argument for config folder: " + config_folder)
    click.echo("Argument for destination package folder: " + dest_folder)
    click.echo("Argument for ignore lists: " + ignore)

    if not platform_is_valid(platform):
        return

    if type == "client" or type == "server":
        click.echo(
            "Now, you are building the fedml packages which will be used in the MLOps "
            "platform."
        )
        click.echo(
            "The packages will be used for client training and server aggregation."
        )
        click.echo(
            "When the building process is completed, you will find the packages in the directory as follows: "
            + os.path.join(dest_folder, "dist-packages")
            + "."
        )
        click.echo(
            "Then you may upload the packages on the configuration page in the MLOps platform to start the "
            "federated learning flow."
        )
        click.echo("Building...")
    else:
        click.echo("You should specify the type argument value as client or server.")
        exit(-1)

    home_dir = expanduser("~")
    mlops_build_path = os.path.join(home_dir, "fedml-mlops-build")
    try:
        shutil.rmtree(mlops_build_path, ignore_errors=True)
    except Exception as e:
        pass

    ignore_list = "{},{}".format(ignore, FEDML_MLOPS_BUILD_PRE_IGNORE_LIST)
    pip_source_dir = os.path.dirname(__file__)
    pip_source_dir = os.path.dirname(pip_source_dir)
    pip_build_path = os.path.join(pip_source_dir, "computing", "scheduler", "build-package")
    build_dir_ignore = "__pycache__,*.pyc,*.git"
    build_dir_ignore_list = tuple(build_dir_ignore.split(','))
    shutil.copytree(pip_build_path, mlops_build_path,
                    ignore_dangling_symlinks=True, ignore=shutil.ignore_patterns(*build_dir_ignore_list))

    if type == "client":
        result = build_mlops_package(
            ignore_list,
            source_folder,
            entry_point,
            config_folder,
            dest_folder,
            mlops_build_path,
            "fedml-client",
            "client-package",
            "${FEDSYS.CLIENT_INDEX}",
        )
        if result != 0:
            exit(result)
        click.echo("You have finished all building process. ")
        click.echo(
            "Now you may use "
            + os.path.join(dest_folder, "dist-packages", "client-package.zip")
            + " to start your federated "
              "learning run."
        )
    elif type == "server":
        result = build_mlops_package(
            ignore_list,
            source_folder,
            entry_point,
            config_folder,
            dest_folder,
            mlops_build_path,
            "fedml-server",
            "server-package",
            "0",
        )
        if result != 0:
            exit(result)

        click.echo("You have finished all building process. ")
        click.echo(
            "Now you may use "
            + os.path.join(dest_folder, "dist-packages", "server-package.zip")
            + " to start your federated "
              "learning run."
        )


def build_mlops_package(
        ignore,
        source_folder,
        entry_point,
        config_folder,
        dest_folder,
        mlops_build_path,
        mlops_package_parent_dir,
        mlops_package_name,
        rank,
):
    if not os.path.exists(source_folder):
        click.echo("source folder is not exist: " + source_folder)
        return -1

    if not os.path.exists(os.path.join(source_folder, entry_point)):
        click.echo(
            "entry file: "
            + entry_point
            + " is not exist in the source folder: "
            + source_folder
        )
        return -1

    if not os.path.exists(config_folder):
        click.echo("config folder is not exist: " + source_folder)
        return -1

    mlops_src = source_folder
    mlops_src_entry = entry_point
    mlops_conf = config_folder
    cur_dir = mlops_build_path
    mlops_package_base_dir = os.path.join(
        cur_dir, "mlops-core", mlops_package_parent_dir
    )
    package_dir = os.path.join(mlops_package_base_dir, mlops_package_name)
    fedml_dir = os.path.join(package_dir, "fedml")
    mlops_dest = fedml_dir
    mlops_dest_conf = os.path.join(fedml_dir, "config")
    mlops_pkg_conf = os.path.join(package_dir, "conf", "fedml.yaml")
    mlops_dest_entry = os.path.join("fedml", mlops_src_entry)
    mlops_package_file_name = mlops_package_name + ".zip"
    dist_package_dir = os.path.join(dest_folder, "dist-packages")
    dist_package_file = os.path.join(dist_package_dir, mlops_package_file_name)
    ignore_list = tuple(ignore.split(','))

    shutil.rmtree(mlops_dest_conf, ignore_errors=True)
    shutil.rmtree(mlops_dest, ignore_errors=True)
    try:
        shutil.copytree(mlops_src, mlops_dest, copy_function=shutil.copy,
                        ignore_dangling_symlinks=True, ignore=shutil.ignore_patterns(*ignore_list))
    except Exception as e:
        pass
    try:
        shutil.copytree(mlops_conf, mlops_dest_conf, copy_function=shutil.copy,
                        ignore_dangling_symlinks=True, ignore=shutil.ignore_patterns(*ignore_list))
    except Exception as e:
        pass
    try:
        os.remove(os.path.join(mlops_dest_conf, "mqtt_config.yaml"))
        os.remove(os.path.join(mlops_dest_conf, "s3_config.yaml"))
    except Exception as e:
        pass

    mlops_pkg_conf_file = open(mlops_pkg_conf, mode="w")
    mlops_pkg_conf_file.writelines(
        [
            "entry_config: \n",
            "  entry_file: " + mlops_dest_entry + "\n",
            "  conf_file: config/fedml_config.yaml\n",
            "dynamic_args:\n",
            "  rank: " + rank + "\n",
            "  run_id: ${FEDSYS.RUN_ID}\n",
            # "  data_cache_dir: ${FEDSYS.PRIVATE_LOCAL_DATA}\n",
            # "  data_cache_dir: /fedml/fedml-package/fedml/data\n",
            "  mqtt_config_path: /fedml/fedml_config/mqtt_config.yaml\n",
            "  s3_config_path: /fedml/fedml_config/s3_config.yaml\n",
            "  log_file_dir: /fedml/fedml-package/fedml/data\n",
            "  log_server_url: ${FEDSYS.LOG_SERVER_URL}\n",
            "  client_id_list: ${FEDSYS.CLIENT_ID_LIST}\n",
            "  client_objects: ${FEDSYS.CLIENT_OBJECT_LIST}\n",
            "  is_using_local_data: ${FEDSYS.IS_USING_LOCAL_DATA}\n",
            "  synthetic_data_url: ${FEDSYS.SYNTHETIC_DATA_URL}\n",
            "  client_num_in_total: ${FEDSYS.CLIENT_NUM}\n",
        ]
    )
    mlops_pkg_conf_file.flush()
    mlops_pkg_conf_file.close()

    local_mlops_package = os.path.join(mlops_package_base_dir, mlops_package_file_name)
    if os.path.exists(local_mlops_package):
        os.remove(os.path.join(mlops_package_base_dir, mlops_package_file_name))
    mlops_archive_name = os.path.join(mlops_package_base_dir, mlops_package_name)
    shutil.make_archive(
        mlops_archive_name,
        "zip",
        root_dir=mlops_package_base_dir,
        base_dir=mlops_package_name,
    )
    if not os.path.exists(dist_package_dir):
        os.makedirs(dist_package_dir, exist_ok=True)
    if os.path.exists(dist_package_file) and not os.path.isdir(dist_package_file):
        os.remove(dist_package_file)
    mlops_archive_zip_file = mlops_archive_name + ".zip"
    if os.path.exists(mlops_archive_zip_file):
        shutil.move(mlops_archive_zip_file, dist_package_file)

    shutil.rmtree(mlops_build_path, ignore_errors=True)

    return 0


@cli.command("diagnosis", help="Diagnosis for open.fedml.ai, AWS S3 service and MQTT service")
@click.help_option("--help", "-h")
@click.option(
    "--open", "-o", default=None, is_flag=True, help="check the connection to open.fedml.ai.",
)
@click.option(
    "--s3", "-s", default=None, is_flag=True, help="check the connection to AWS S3 server.",
)
@click.option(
    "--mqtt", "-m", default=None, is_flag=True, help="check the connection to mqtt.fedml.ai (1883).",
)
@click.option(

    "--mqtt_daemon", "-d", default=None, is_flag=True,
    help="check the connection to mqtt.fedml.ai (1883) with loop mode.",
)
@click.option(
    "--mqtt_s3_backend_server", "-msbs", default=None, is_flag=True,
    help="check the connection to mqtt.fedml.ai (1883) as mqtt+s3 server.",
)
@click.option(
    "--mqtt_s3_backend_client", "-msbc", default=None, is_flag=True,
    help="check the connection to mqtt.fedml.ai (1883) as mqtt+s3 client.",
)
@click.option(
    "--mqtt_s3_backend_run_id", "-rid", type=str, default="fedml_diag_9988", help="mqtt+s3 run id.",
)
def mlops_diagnosis(open, s3, mqtt, mqtt_daemon, mqtt_s3_backend_server, mqtt_s3_backend_client,
                    mqtt_s3_backend_run_id):
    check_open = open
    check_s3 = s3
    check_mqtt = mqtt
    check_mqtt_daemon = mqtt_daemon

    check_mqtt_s3_backend_server = mqtt_s3_backend_server
    check_mqtt_s3_backend_client = mqtt_s3_backend_client
    run_id = mqtt_s3_backend_run_id

    if open is None and s3 is None and mqtt is None:
        check_open = True
        check_s3 = True
        check_mqtt = True

    if mqtt_daemon is None:
        check_mqtt_daemon = False

    if mqtt_s3_backend_server is None:
        check_mqtt_s3_backend_server = False

    if mqtt_s3_backend_client is None:
        check_mqtt_s3_backend_client = False

    if check_open:
        is_open_connected = ClientDiagnosis.check_open_connection()
        if is_open_connected:
            click.echo("The connection to https://open.fedml.ai is OK.")
        else:
            click.echo("You can not connect to https://open.fedml.ai.")

    if check_s3:
        is_s3_connected = ClientDiagnosis.check_s3_connection()
        if is_s3_connected:
            click.echo("The connection to AWS S3 is OK.")
        else:
            click.echo("You can not connect to AWS S3.")

    if check_mqtt:
        is_mqtt_connected = ClientDiagnosis.check_mqtt_connection()
        if is_mqtt_connected:
            click.echo("The connection to mqtt.fedml.ai (port:1883) is OK.")
        else:
            click.echo("You can not connect to mqtt.fedml.ai (port:1883).")

    if check_mqtt_daemon:
        ClientDiagnosis.check_mqtt_connection_with_daemon_mode()

    sys_utils.cleanup_all_fedml_client_diagnosis_processes()
    if check_mqtt_s3_backend_server:
        pip_source_dir = os.path.dirname(__file__)
        pip_source_dir = os.path.dirname(pip_source_dir)
        server_diagnosis_cmd = os.path.join(pip_source_dir, "computing", "scheduler", "slave", "client_diagnosis.py")
        backend_server_process = sys_utils.run_subprocess_open([
            sys_utils.get_python_program(),
            server_diagnosis_cmd,
            "-t",
            "server",
            "-r",
            run_id
        ]
        ).pid

    if check_mqtt_s3_backend_client:
        pip_source_dir = os.path.dirname(__file__)
        pip_source_dir = os.path.dirname(pip_source_dir)
        client_diagnosis_cmd = os.path.join(pip_source_dir, "computing", "scheduler", "slave", "client_diagnosis.py")
        backend_client_process = sys_utils.run_subprocess_open([
            sys_utils.get_python_program(),
            client_diagnosis_cmd,
            "-t",
            "client",
            "-r",
            run_id
        ]
        ).pid


@cli.command(
    "env",
    help="collect the environment information to help debugging, including OS, Hardware Architecture, "
         "Python version, etc.",
)
@click.help_option("--help", "-h")
def env():
    collect_env()


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
    # Show job info
    list_jobs_core(platform, None, None, job_id[0], api_key, version)

    # Get job logs
    FedMLJobManager.get_instance().set_config_version(version)
    job_logs = FedMLJobManager.get_instance().get_job_logs(job_id[0], 1, Constants.JOB_LOG_PAGE_SIZE, api_key)

    # Show job log summary info
    log_head_table = PrettyTable(['Job ID', 'Total Log Lines', 'Log URL'])
    log_head_table.add_row([job_id[0], job_logs.total_num, job_logs.log_full_url])
    click.echo("\nLogs summary info is as follows.")
    print(log_head_table)

    # Show job logs URL for each device
    if len(job_logs.log_devices) > 0:
        log_device_table = PrettyTable(['Device ID', 'Device Name', 'Device Log URL'])
        for log_device in job_logs.log_devices:
            log_device_table.add_row([log_device.device_id, log_device.device_name, log_device.log_url])
        click.echo("\nLogs URL for each device is as follows.")
        print(log_device_table)

    # Show job log lines
    if len(job_logs.log_lines):
        click.echo("\nAll logs is as follows.")
        for log_line in job_logs.log_lines:
            click.echo(str(log_line).rstrip('\n'))

        for page_count in range(2, job_logs.total_pages+1):
            job_logs = FedMLJobManager.get_instance().get_job_logs(job_id[0], page_count,
                                                                   Constants.JOB_LOG_PAGE_SIZE, api_key)
            for log_line in job_logs.log_lines:
                click.echo(str(log_line).rstrip('\n'))


@launch.command("queue", help="View the job queue at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("group_id", nargs=-1)
def launch_queue(group_id):
    pass


@launch.command(
    "run", help="Launch job at the FedML® Launch platform (open.fedml.ai)",
    context_settings={"ignore_unknown_options": True}
)
@click.help_option("--help", "-h")
@click.argument("yaml_file", nargs=-1)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--group",
    "-g",
    type=str,
    default="",
    help="The queue group id on which your job will be scheduled.",
)
@click.option(
    "--devices_server", "-ds", type=str, default="",
    help="The server to run the launching job, for the launch platform, we do not need to set this option."
)
@click.option(
    "--devices_edges", "-de", type=str, default="",
    help="The edge devices to run the launching job. Seperated with ',', e.g. 705,704. "
         "for the launch platform, we do not need to set this option."
)
@click.option(
    "--no_confirmation", "-nc", default=None, is_flag=True,
    help="allow the Launch platform to select compute resource without confirmation after initiating launching request.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="launch job to which version of MLOps platform. It should be dev, test or release",
)
def launch_job(yaml_file, api_key, platform, group,
               devices_server, devices_edges, no_confirmation, version):
    if not platform_is_valid(platform):
        return

    is_no_confirmation = no_confirmation
    if no_confirmation is None:
        is_no_confirmation = False

    if api_key is None or api_key == "":
        saved_api_key = FedMLLaunchManager.get_api_key()
        if saved_api_key is None or saved_api_key == "":
            api_key = click.prompt("FedML® Launch API Key is not set yet, please input your API key", hide_input=True)
        else:
            api_key = saved_api_key

    FedMLLaunchManager.get_instance().set_config_version(version)
    is_valid_heartbeat = FedMLLaunchManager.get_instance().check_heartbeat(api_key)
    if not is_valid_heartbeat:
        click.echo("Your API Key is not correct. Please input again.")
        api_key = click.prompt("FedML® Launch API Key is not set yet, please input your API key", hide_input=True)
        is_valid_heartbeat = FedMLLaunchManager.get_instance().check_heartbeat(api_key)
        if not is_valid_heartbeat:
            click.echo("Your API Key is not correct. Please check and try again.")
            return
    if is_valid_heartbeat:
        FedMLLaunchManager.save_api_key(api_key)

    FedMLLaunchManager.get_instance().set_config_version(version)
    result = FedMLLaunchManager.get_instance().launch_job(yaml_file[0], api_key,
                                                          platform,
                                                          devices_server, devices_edges,
                                                          no_confirmation=is_no_confirmation)
    if result is not None:
        if result.status == Constants.JOB_START_STATUS_INVALID:
            click.echo(f"\nPlease check your {os.path.basename(yaml_file[0])} file "
                       f"to make sure the syntax is valid, e.g. "
                       f"whether minimum_num_gpus or maximum_cost_per_hour is valid.")
            return
        elif result.status == Constants.JOB_START_STATUS_BLOCKED:
            click.echo("\nBecause the value of maximum_cost_per_hour is too low,"
                       "we can not find exactly matched machines for your job. \n"
                       "But here we still present machines closest to your expected price as below.")
        elif result.status == Constants.JOB_START_STATUS_QUEUED:
            click.echo("\nNo resource available now, but we can keep your job in the waiting queue.")
            if click.confirm("Do you want to join the queue?", abort=False):
                click.echo("You have confirmed to keep your job in the waiting list.")
                return
            else:
                stop_jobs_core(platform, result.job_id, api_key, version)
                return
        elif result.status == Constants.JOB_START_STATUS_BIND_CREDIT_CARD_FIRST:
            click.echo("Please bind your credit card before launching the job.")
            return
        elif result.status == Constants.JOB_START_STATUS_QUERY_CREDIT_CARD_BINDING_STATUS_FAILED:
            click.echo("Failed to query credit card binding status. Please try again later.")
            return

        if result.job_url == "":
            if result.message is not None:
                click.echo(f"Failed to launch the job with response messages: {result.message}")
            else:
                click.echo(f"Failed to launch the job. Please check if the network is available "
                           f"or the job name {result.job_name} is duplicated.")
        else:
            if is_no_confirmation:
                click.echo("Job{}has started.".format(f" {result.job_name} " if result.job_name is not None else " "))
                if result.job_url is not None:
                    click.echo(f"Please go to this web page with your account "
                               f"to review your job details.")
                    click.echo(f"{result.job_url}")

                if hasattr(result, "gpu_matched") and result.gpu_matched is not None and len(result.gpu_matched) > 0:
                    if result.status != Constants.JOB_START_STATUS_BLOCKED:
                        click.echo(f"\nSearched and matched the following GPU resource for your job:")
                    gpu_table = PrettyTable(['Provider', 'Instance', 'vCPU(s)', 'Memory(GB)', 'GPU(s)',
                                             'Region', 'Cost', 'Selected'])
                    for gpu_device in result.gpu_matched:
                        gpu_table.add_row([gpu_device.gpu_provider, gpu_device.gpu_instance, gpu_device.cpu_count,
                                           gpu_device.mem_size,
                                           f"{gpu_device.gpu_type}:{gpu_device.gpu_num}",
                                           gpu_device.gpu_region, gpu_device.cost, Constants.CHECK_MARK_STRING])
                    print(gpu_table)
                    click.echo("")
            else:
                if hasattr(result, "gpu_matched") and result.gpu_matched is not None and len(result.gpu_matched) > 0:
                    if result.status != Constants.JOB_START_STATUS_BLOCKED:
                        click.echo(f"\nSearched and matched the following GPU resource for your job:")
                    gpu_table = PrettyTable(['Provider', 'Instance', 'vCPU(s)', 'Memory(GB)', 'GPU(s)',
                                             'Region', 'Cost', 'Selected'])
                    for gpu_device in result.gpu_matched:
                        gpu_table.add_row([gpu_device.gpu_provider, gpu_device.gpu_instance, gpu_device.cpu_count,
                                           gpu_device.mem_size,
                                           f"{gpu_device.gpu_type}:{gpu_device.gpu_num}",
                                           gpu_device.gpu_region, gpu_device.cost, Constants.CHECK_MARK_STRING])
                    print(gpu_table)
                    click.echo("")

                    if result.job_url is not None:
                        click.echo(f"You can also view the matched GPU resource with Web UI at: ")
                        click.echo(f"{result.job_url}")

                    click.echo("")
                    if click.confirm(f"Are you sure to launch it?", abort=False):
                        click.echo("")
                        result = FedMLLaunchManager.get_instance().start_job(
                            platform, result.project_name, result.application_name,
                            devices_server, devices_edges, api_key,
                            no_confirmation=True, job_id=result.job_id)
                        if result is not None:
                            if result.job_url == "":
                                if result.message is not None:
                                    click.echo(f"Failed to launch the job with response messages: {result.message}")
                            else:
                                FedMLJobManager.get_instance().set_config_version(version)
                                job_list_obj = FedMLJobManager.get_instance().list_job(platform, result.project_name,
                                                                                       None, api_key,
                                                                                       job_id=result.job_id)
                                if job_list_obj is not None:
                                    if len(job_list_obj.job_list) > 0:
                                        if len(job_list_obj.job_list) > 0:
                                            click.echo("Your launch result is as follows:")
                                        jobs_count = 0
                                        job_list_table = PrettyTable(['Job Name', 'Job ID', 'Status', 'Created',
                                                                      'Spend Time(hour)', 'Cost'])
                                        for job in job_list_obj.job_list:
                                            jobs_count += 1
                                            job_list_table.add_row(
                                                [job.job_name, job.job_id, job.status, job.started_time,
                                                 job.compute_duration, job.cost])
                                        print(job_list_table)
                                click.echo("\nYou can track your job running details at this URL:")
                                click.echo(f"{result.job_url}")
                        else:
                            click.echo(f"Failed to launch the job.")
                    else:
                        stop_jobs_core(platform, result.job_id, api_key, version, show_hint_texts=False)
                        return
                else:
                    click.echo(f"Result of launching job: code={result.status}, message=\"{result.message}\".")
                    return

            if result is not None:
                click.echo("")
                click.echo(f"For querying the realtime status of your job, please run the following command.")
                click.echo(f"fedml launch log {result.job_id}" +
                           "{}".format(f" -v {version}" if version == "dev" else ""))
    else:
        click.echo(f"Failed to launch the job.")


@cli.group("jobs")
def jobs():
    """
    Manage jobs on the MLOps platform.
    """
    pass


@jobs.command("start", help="Start a job at the MLOps platform.")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="octopus",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch."
)
@click.option(
    "--project_name",
    "-prj",
    type=str,
    help="The project name at the MLOps platform.",
)
@click.option(
    "--application_name",
    "-app",
    type=str,
    help="Application name in the My Application list at the MLOps platform.",
)
@click.option(
    "--job_name",
    "-jn",
    type=str,
    default="",
    help="The job name at the MLOps platform.",
)
@click.option(
    "--devices_server", "-ds", type=str, default="",
    help="The server to run the launching job, for the launch platform, we do not need to set this option."
)
@click.option(
    "--devices_edges", "-de", type=str, default="",
    help="The edge devices to run the launching job. Seperated with ',', e.g. 705,704. "
         "for the launch platform, we do not need to set this option."
)
@click.option(
    "--user", "-u", type=str, help="user id.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="start job at which version of MLOps platform. It should be dev, test or release",
)
def start_job(platform, project_name, application_name, job_name, devices_server, devices_edges, user, api_key,
              version):
    if not platform_is_valid(platform):
        return

    if api_key is None or api_key == "":
        saved_api_key = FedMLLaunchManager.get_api_key()
        if saved_api_key is None or saved_api_key == "":
            api_key = click.prompt("FedML® Launch API Key is not set yet, please input your API key", hide_input=True)
        else:
            api_key = saved_api_key

    FedMLLaunchManager.get_instance().set_config_version(version)
    is_valid_heartbeat = FedMLLaunchManager.get_instance().check_heartbeat(api_key)
    if not is_valid_heartbeat:
        click.echo("Your API Key is not correct. Please input again.")
        api_key = click.prompt("FedML® Launch API Key is not set yet, please input your API key", hide_input=True)
        is_valid_heartbeat = FedMLLaunchManager.get_instance().check_heartbeat(api_key)
        if not is_valid_heartbeat:
            click.echo("Your API Key is not correct. Please check and try again.")
            return
    if is_valid_heartbeat:
        FedMLLaunchManager.save_api_key(api_key)

    FedMLJobManager.get_instance().set_config_version(version)
    result = FedMLJobManager.get_instance().start_job(platform, project_name, application_name,
                                                      devices_server, devices_edges,
                                                      user, api_key,
                                                      job_name=job_name, need_confirmation=False)
    if result:
        click.echo(f"Job {result.job_name} has started.")
        click.echo(f"Please go to this web page with your account id {result.user_id} to review your job details.")
        click.echo(f"{result.job_url}")
        click.echo(f"For querying the status of the job, please run the command: "
                   f"fedml jobs list -id {result.job_id}")
    else:
        click.echo("Failed to start job, please check your network connection "
                   "and make sure be able to access the MLOps platform.")


@jobs.command("list", help="List jobs from the MLOps platform.")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--project_name",
    "-prj",
    type=str,
    help="The project name at the MLOps platform.",
)
@click.option(
    "--job_name",
    "-n",
    type=str,
    help="Job name at the MLOps platform.",
)
@click.option(
    "--job_id",
    "-id",
    type=str,
    default="",
    help="Job id at the MLOps platform.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="list jobs at which version of MLOps platform. It should be dev, test or release",
)
def list_jobs(platform, project_name, job_name, job_id, api_key, version):
    list_jobs_core(platform, project_name, job_name, job_id, api_key, version)


def list_jobs_core(platform, project_name, job_name, job_id, api_key, version):
    if not platform_is_valid(platform):
        return

    if api_key is None or api_key == "":
        saved_api_key = FedMLLaunchManager.get_api_key()
        if saved_api_key is None or saved_api_key == "":
            api_key = click.prompt("FedML® Launch API Key is not set yet, please input your API key", hide_input=True)
        else:
            api_key = saved_api_key

    FedMLLaunchManager.get_instance().set_config_version(version)
    is_valid_heartbeat = FedMLLaunchManager.get_instance().check_heartbeat(api_key)
    if not is_valid_heartbeat:
        click.echo("Your API Key is not correct. Please input again.")
        api_key = click.prompt("FedML® Launch API Key is not set yet, please input your API key", hide_input=True)
        is_valid_heartbeat = FedMLLaunchManager.get_instance().check_heartbeat(api_key)
        if not is_valid_heartbeat:
            click.echo("Your API Key is not correct. Please check and try again.")
            return
    if is_valid_heartbeat:
        FedMLLaunchManager.save_api_key(api_key)

    FedMLJobManager.get_instance().set_config_version(version)
    job_list_obj = FedMLJobManager.get_instance().list_job(platform, project_name, job_name,
                                                           api_key, job_id=job_id)
    if job_list_obj is not None:
        if len(job_list_obj.job_list) > 0:
            if len(job_list_obj.job_list) > 0:
                click.echo("Found the following matched jobs.")
            job_list_table = PrettyTable(['Job Name', 'Job ID', 'Status', 'Created', 'Spend Time(hour)', 'Cost'])
            jobs_count = 0
            for job in job_list_obj.job_list:
                jobs_count += 1
                device_count = 0
                device_list = ""
                for device_info_item in job.device_infos:
                    device_count += 1
                    device_list += f"({device_count}). {device_info_item} "

                job_list_table.add_row([job.job_name, job.job_id, job.status, job.started_time,
                                        job.compute_duration, job.cost])

            print(job_list_table)
        else:
            click.echo("Not found any jobs.")
    else:
        click.echo("Failed to list jobs, please check your network connection "
                   "and make sure be able to access the MLOps platform.")


@jobs.command("stop", help="Stop a job from the MLOps platform.")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the MLOps platform (options: octopus, parrot, spider, beehive, falcon, launch, "
         "default is falcon).",
)
@click.option(
    "--job_id",
    "-id",
    type=str,
    default="",
    help="Job id at the MLOps platform.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="stop a job at which version of MLOps platform. It should be dev, test or release",
)
def stop_jobs(platform, job_id, api_key, version):
    stop_jobs_core(platform, job_id, api_key, version)


def stop_jobs_core(platform, job_id, api_key, version, show_hint_texts=True):
    if not platform_is_valid(platform):
        return

    FedMLJobManager.get_instance().set_config_version(version)
    is_stopped = FedMLJobManager.get_instance().stop_job(platform, job_id, api_key)
    if show_hint_texts:
        if is_stopped:
            click.echo("Job has been stopped.")
        else:
            click.echo("Failed to stop the job, please check your network connection "
                       "and make sure be able to access the MLOps platform.")
    return is_stopped


@cli.group("model")
def model():
    """
    Deploy and infer models.
    """
    pass


@model.group("device")
def device():
    """
    Manage computing device.
    """
    pass


@device.command("login", help="Login as model device agent(MDA) on the ModelOps platform (open.fedml.ai).")
@click.help_option("--help", "-h")
@click.argument("userid", nargs=-1)
@click.option(
    "--cloud", "-c", default=None, is_flag=True, help="login as fedml cloud device.",
)
@click.option(
    "--on_premise", "-p", default=None, is_flag=True, help="login as on-premise device.",
)
@click.option(
    "--master", "-m", default=None, is_flag=True, help="login as master device in the federated inference cluster.",
)
@click.option(
    "--infer_host", "-ih", type=str, default="127.0.0.1",
    help="used this ip address or domain name as inference host.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="login to which version of ModelOps platform. It should be dev, test or release",
)
@click.option(
    "--local_server",
    "-ls",
    type=str,
    default="127.0.0.1",
    help="local server address.",
)
@click.option(
    "--runner_cmd",
    "-rc",
    type=str,
    default="{}",
    help="runner commands (options: request json for starting deployment, stopping deployment).",
)
@click.option(
    "--device_id", "-id", type=str, default="0", help="device id.",
)
@click.option(
    "--os_name", "-os", type=str, default="", help="os name.",
)
@click.option(
    "--docker", "-d", default=None, is_flag=True, help="login with docker mode at the model device agent.",
)
@click.option(
    "--docker-rank", "-dr", default="1", help="docker client rank index (from 1 to n).",
)
@click.option(
    "--redis_addr", "-ra", default="local", help="redis addr for caching inference information in the master device.",
)
@click.option(
    "--redis_port", "-rp", default="6379", help="redis port for caching inference information in the master device.",
)
@click.option(
    "--redis_password", "-rpw", default="fedml_default",
    help="redis password for caching inference information in the master device.",
)
def login_as_model_device_agent(
        userid, cloud, on_premise, master, infer_host, version, local_server,
        runner_cmd, device_id, os_name, docker, docker_rank, redis_addr, redis_port, redis_password
):
    device_login_entry.login_as_model_device_agent(userid, cloud, on_premise, master, infer_host, version, local_server,
                                                   runner_cmd, device_id, os_name, docker, docker_rank,
                                                   redis_addr, redis_port, redis_password)


@device.command("logout", help="Logout from the ModelOps platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.option(
    "--slave", "-s", default=None, is_flag=True, help="logout from slave device.",
)
@click.option(
    "--master", "-m", default=None, is_flag=True, help="logout from master device.",
)
@click.option(
    "--docker", "-d", default=None, is_flag=True, help="logout from docker mode at the model device agent.",
)
@click.option(
    "--docker-rank", "-dr", default=None, help="docker client rank index (from 1 to n).",
)
def logout_from_model_ops(slave, master, docker, docker_rank):
    device_login_entry.logout_from_model_ops(slave, master, docker, docker_rank)
    print("\nlogout successfully!\n")


@model.command("create", help="Create local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def create_model(name):
    if FedMLModelCards.get_instance().create_model(name):
        click.echo("Create model {} successfully.".format(name))
    else:
        click.echo("Failed to create model {}.".format(name))


@model.command("delete", help="Delete local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def delete_model(name):
    if FedMLModelCards.get_instance().delete_model(name):
        click.echo("Delete model {} successfully.".format(name))
    else:
        click.echo("Failed to delete model {}.".format(name))


@model.command("add", help="Add file to local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--path", "-p", type=str, help="path for specific model.",
)
def add_model_files(name, path):
    if FedMLModelCards.get_instance().add_model_files(name, path):
        click.echo("Add file to model {} successfully.".format(name))
    else:
        click.echo("Failed to add file to model {}.".format(name))


@model.command("remove", help="Remove file from local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--file", "-f", type=str, help="file name for specific model.",
)
def remove_model_files(name, file):
    if FedMLModelCards.get_instance().remove_model_files(name, file):
        click.echo("Remove file from model {} successfully.".format(name))
    else:
        click.echo("Failed to remove file from model {}.".format(name))


@model.command("list", help="List model in the local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def list_models(name):
    models = FedMLModelCards.get_instance().list_models(name)
    if len(models) <= 0:
        click.echo("Model list is empty.")
    else:
        for model_item in models:
            click.echo(model_item)
        click.echo("List model {} successfully.".format(name))


@model.command("list-remote", help="List models in the remote model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--user", "-u", type=str, help="user id.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="interact with which version of ModelOps platform. It should be dev, test or release",
)
@click.option(
    "--local_server",
    "-ls",
    type=str,
    default="127.0.0.1",
    help="local server address.",
)
def list_remote_models(name, user, api_key, version, local_server):
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    FedMLModelCards.get_instance().set_config_version(version)
    model_query_result = FedMLModelCards.get_instance().list_models(name, user, api_key, local_server)
    if model_query_result is None or model_query_result.model_list is None or len(model_query_result.model_list) <= 0:
        click.echo("Model list is empty.")
    else:
        click.echo("Found {} models:".format(len(model_query_result.model_list)))
        index = 1
        for model_item in model_query_result.model_list:
            model_item.show("{}. ".format(index))
            index += 1
        click.echo("List model {} successfully.".format(name))


@model.command("package", help="Build local model repository as zip model package.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def package_model(name):
    model_zip = FedMLModelCards.get_instance().build_model(name)
    if model_zip != "":
        click.echo("Build model package {} successfully".format(name))
        click.echo("The local package is located at {}.".format(model_zip))
    else:
        click.echo("Failed to build model {}.".format(name))


@model.command("push", help="Push local model repository to ModelOps(open.fedml.ai).")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--model_storage_url", "-s", type=str, help="model storage url.",
)
@click.option(
    "--model_net_url", "-mn", type=str, help="model net url.",
)
@click.option(
    "--user", "-u", type=str, help="user id.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="interact with which version of ModelOps platform. It should be dev, test or release",
)
@click.option(
    "--local_server",
    "-ls",
    type=str,
    default="127.0.0.1",
    help="local server address.",
)
def push_model(name, model_storage_url, model_net_url, user, api_key, version, local_server):
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    FedMLModelCards.get_instance().set_config_version(version)
    model_is_from_open = True if model_storage_url is not None and model_storage_url != "" else False
    model_storage_url, model_zip = FedMLModelCards.get_instance().push_model(name, user, api_key,
                                                                             model_storage_url=model_storage_url,
                                                                             model_net_url=model_net_url,
                                                                             local_server=local_server)
    if model_is_from_open:
        click.echo("Push model {} with model storage url {} successfully.".format(name, model_storage_url))
    else:
        if model_storage_url != "":
            click.echo("Push model {} successfully".format(name))
            click.echo("The remote model storage is located at {}".format(model_storage_url))
            click.echo("The local model package is locate at .".format(model_zip))
        else:
            click.echo("Failed to push model {}.".format(name))


@model.command("pull", help="Pull remote model(ModelOps) to local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--user", "-u", type=str, help="user id.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="interact with which version of ModelOps platform. It should be dev, test or release",
)
@click.option(
    "--local_server",
    "-ls",
    type=str,
    default="127.0.0.1",
    help="local server address.",
)
def pull_model(name, user, api_key, version, local_server):
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    FedMLModelCards.get_instance().set_config_version(version)
    if FedMLModelCards.get_instance().pull_model(name, user, api_key, local_server):
        click.echo("Pull model {} successfully.".format(name))
    else:
        click.echo("Failed to pull model {}.".format(name))


@model.command("deploy",
               help="Deploy specific model to ModelOps platform(open.fedml.ai) or just for local debugging deployment.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--on_premise", "-p", default=None, is_flag=True, help="all devices are from on-premise.",
)
@click.option(
    "--cloud", "-c", default=None, is_flag=True, help="all devices are from fedml cloud.",
)
@click.option(
    "--devices", "-d", type=str, help="device list, format: [1,2,3]. The first id is master device.",
)
@click.option(
    "--user", "-u", type=str, help="user id or api key.",
)
@click.option(
    "--api_key", "-k", type=str, help="user api key.",
)
@click.option(
    "--params", "-pa", type=str, default="", help="serving parameters.",
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="interact with which version of ModelOps platform. It should be dev, test or release",
)
@click.option(
    "--local_server",
    "-ls",
    type=str,
    default="127.0.0.1",
    help="local server address.",
)
@click.option(
    "--use_local_deployment", "-ld", default=None, is_flag=True,
    help="deploy local model repository by sending MQTT message(just use for debugging).",
)
def deploy_model(name, on_premise, cloud, devices, user, api_key, params, version,
                 local_server, use_local_deployment):
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return

    is_cloud = cloud
    is_on_premise = on_premise
    if cloud is None and on_premise is None:
        is_on_premise = True
    if is_cloud and is_on_premise:
        is_cloud = False

    is_local_dev = use_local_deployment
    if use_local_deployment is None:
        is_local_dev = False

    if is_on_premise:
        device_type = "md.on_premise_device"
    else:
        device_type = "md.fedml_cloud_device"
    FedMLModelCards.get_instance().set_config_version(version)

    params_dict = {}
    if is_local_dev:
        params_dict = json.loads(params)  # load config from Cli

    if FedMLModelCards.get_instance().deploy_model(name, device_type, devices, user, api_key,
                                                   params_dict, use_local_deployment,
                                                   local_server):
        click.echo("Deploy model {} successfully.".format(name))
    else:
        click.echo("Failed to deploy model {}.".format(name))


@model.group("inference")
def inference():
    """
    Inference models.
    """
    pass


@inference.command("query", help="Query inference parameters for specific model from ModelOps platform(open.fedml.ai).")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def query_model_infer(name):
    inference_output_url, model_metadata, model_config = FedMLModelCards.get_instance().query_model(name)
    if inference_output_url != "":
        click.echo("Query model {} successfully.".format(name))
        click.echo("infer url: {}.".format(inference_output_url))
        click.echo("model metadata: {}.".format(model_metadata))
        click.echo("model config: {}.".format(model_config))
    else:
        click.echo("Failed to query model {}.".format(name))


@inference.command("run", help="Run inference action for specific model from ModelOps platform(open.fedml.ai).")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--data", "-d", type=str, help="input data for model inference.",
)
def run_model_infer(name, data):
    infer_out_json = FedMLModelCards.get_instance().inference_model(name, data)
    if infer_out_json != "":
        click.echo("Inference model {} successfully.".format(name))
        click.echo("Result: {}.".format(infer_out_json))
    else:
        click.echo("Failed to inference model {}.".format(name))


if __name__ == "__main__":
    cli()
