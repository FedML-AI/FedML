import os
import shutil
import subprocess
from os.path import expanduser

import click

import fedml

from ..cli.edge_deployment.client_constants import ClientConstants
from ..cli.server_deployment.server_constants import ServerConstants
from ..cli.edge_deployment.client_login import logout as client_logout
from ..cli.env.collect_env import collect_env
from ..cli.server_deployment.server_login import logout as server_logout
from ..cli.edge_deployment.docker_login import login_with_docker_mode
from ..cli.edge_deployment.docker_login import logout_with_docker_mode
from ..cli.edge_deployment.docker_login import logs_with_docker_mode
from ..cli.server_deployment.docker_login import login_with_server_docker_mode
from ..cli.server_deployment.docker_login import logout_with_server_docker_mode
from ..cli.server_deployment.docker_login import logs_with_server_docker_mode
from ..cli.comm_utils import sys_utils


FEDML_MLOPS_BUILD_PRE_IGNORE_LIST = 'dist-packages,client-package.zip,server-package.zip,__pycache__,*.pyc,*.git'
simulator_process_list = list()


@click.group()
def cli():
    pass


@cli.command("version", help="Display fedml version.")
def mlops_version():
    click.echo("fedml version: " + str(fedml.__version__))


@cli.command("status", help="Display fedml client training status.")
def mlops_status():
    training_infos = ClientConstants.get_training_infos()
    click.echo(
        "Client training status: " + str(training_infos["training_status"]).upper()
    )


@cli.command("logs", help="Display fedml logs.")
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
    run_id, edge_id = sys_utils.get_running_info(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
    home_dir = expanduser("~")
    log_file = "{}/{}/fedml/logs/fedml-run-{}-edge-{}.log".format(
        home_dir, ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME, str(run_id), str(edge_id)
    )
    if os.path.exists(log_file):
        with open(log_file) as file_handle:
            log_lines = file_handle.readlines()
        for log_line in log_lines:
            click.echo(log_line)


def display_server_logs():
    run_id, edge_id = sys_utils.get_running_info(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)
    home_dir = expanduser("~")
    log_file = "{}/{}/fedml/logs/fedml-run-{}-edge-{}.log".format(
        home_dir, ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME, str(run_id), str(edge_id)
    )
    if os.path.exists(log_file):
        with open(log_file) as file_handle:
            log_lines = file_handle.readlines()
        for log_line in log_lines:
            click.echo(log_line)


@cli.command("login", help="Login to MLOps platform")
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
    help="run as the role (options: edge_server, cloud_agent, cloud_server, edge_simulator.",
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
    userid, version, client, server, local_server, role, runner_cmd, device_id, os_name, docker, docker_rank
):
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

    # Check docker mode.
    is_docker = docker
    if docker is None:
        is_docker = False

    # click.echo("login as client: {}, as server: {}".format(is_client, is_server))
    if is_client is True:
        if is_docker:
            login_with_docker_mode(account_id, version, docker_rank)
            return
        pip_source_dir = os.path.dirname(__file__)
        login_cmd = os.path.join(pip_source_dir, "edge_deployment", "client_daemon.py")
        client_logout()
        sys_utils.cleanup_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_client_learning_processes()
        sys_utils.cleanup_all_fedml_client_login_processes("client_login.py")

        try:
            ClientConstants.login_role_list.index(role)
        except ValueError as e:
            role = ClientConstants.login_role_list[ClientConstants.LOGIN_MODE_CLIEN_INDEX]

        login_pid = subprocess.Popen(
            [
                sys_utils.get_python_program(),
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
                os_name
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
        login_cmd = os.path.join(pip_source_dir, "server_deployment", "server_daemon.py")
        server_logout()
        sys_utils.cleanup_login_process(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_server_learning_processes()
        sys_utils.cleanup_all_fedml_server_login_processes("server_login.py")
        login_pid = subprocess.Popen(
            [
                sys_utils.get_python_program(),
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
                os_name
            ]
        ).pid
        sys_utils.save_login_process(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME,
                                     ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME, login_pid)


@cli.command("logout", help="Logout from MLOps platform (open.fedml.ai)")
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
        client_logout()
        sys_utils.cleanup_login_process(ClientConstants.LOCAL_HOME_RUNNER_DIR_NAME, ClientConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_client_learning_processes()
        sys_utils.cleanup_all_fedml_client_login_processes("client_login.py")

    if is_server is True:
        if is_docker:
            logout_with_server_docker_mode(docker_rank)
            return
        server_logout()
        sys_utils.cleanup_login_process(ServerConstants.LOCAL_HOME_RUNNER_DIR_NAME, ServerConstants.LOCAL_RUNNER_INFO_DIR_NAME)
        sys_utils.cleanup_all_fedml_server_learning_processes()
        sys_utils.cleanup_all_fedml_server_login_processes("server_login.py")


@cli.command("build", help="Build packages for MLOps platform (open.fedml.ai)")
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
def mlops_build(type, source_folder, entry_point, config_folder, dest_folder, ignore):
    click.echo("Argument for type: " + type)
    click.echo("Argument for source folder: " + source_folder)
    click.echo("Argument for entry point: " + entry_point)
    click.echo("Argument for config folder: " + config_folder)
    click.echo("Argument for destination package folder: " + dest_folder)
    click.echo("Argument for ignore lists: " + ignore)

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
    pip_build_path = os.path.join(pip_source_dir, "build-package")
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
        try:
            os.makedirs(dist_package_dir)
        except Exception as e:
            pass
    if os.path.exists(dist_package_file) and not os.path.isdir(dist_package_file):
        os.remove(dist_package_file)
    mlops_archive_zip_file = mlops_archive_name + ".zip"
    if os.path.exists(mlops_archive_zip_file):
        shutil.move(mlops_archive_zip_file, dist_package_file)

    shutil.rmtree(mlops_build_path, ignore_errors=True)

    return 0


@cli.command(
    "env",
    help="collect the environment information to help debugging, including OS, Hardware Architecture, "
    "Python version, etc.",
)
def env():
    collect_env()


@cli.command(
    "launch", help="launch tool", context_settings={"ignore_unknown_options": True}
)
@click.argument("arguments", nargs=-1, type=click.Path())
def launch(arguments):
    # for argument in arguments:
    #     click.echo(argument)

    from fedml.cross_silo.client.client_launcher import CrossSiloLauncher
    CrossSiloLauncher.launch_dist_trainers(arguments[0], list(arguments[1:]))


if __name__ == "__main__":
    cli()
