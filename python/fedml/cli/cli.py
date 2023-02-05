import os
import pickle
import shutil
import subprocess
from os.path import expanduser

import click

import fedml

# from ..cli.edge_deployment.client_constants import ClientConstants
# from ..cli.server_deployment.server_constants import ServerConstants
# from ..cli.edge_deployment.client_login import logout as client_logout
# from ..cli.env.collect_env import collect_env
# from ..cli.server_deployment.server_login import logout as server_logout
# from ..cli.edge_deployment.docker_login import login_with_docker_mode
# from ..cli.edge_deployment.docker_login import logout_with_docker_mode
# from ..cli.edge_deployment.docker_login import logs_with_docker_mode
# from ..cli.server_deployment.docker_login import login_with_server_docker_mode
# from ..cli.server_deployment.docker_login import logout_with_server_docker_mode
# from ..cli.server_deployment.docker_login import logs_with_server_docker_mode
# from ..cli.edge_deployment.client_diagnosis import ClientDiagnosis
# from ..cli.comm_utils import sys_utils
# from .model_deployment import device_login_entry
# from .model_deployment.device_model_cards import FedMLModelCards
import torch

from fedml.cli.edge_deployment.client_constants import ClientConstants
from fedml.cli.server_deployment.server_constants import ServerConstants
from fedml.cli.edge_deployment.client_login import logout as client_logout
from fedml.cli.env.collect_env import collect_env
from fedml.cli.server_deployment.server_login import logout as server_logout
from fedml.cli.edge_deployment.docker_login import login_with_docker_mode
from fedml.cli.edge_deployment.docker_login import logout_with_docker_mode
from fedml.cli.edge_deployment.docker_login import logs_with_docker_mode
from fedml.cli.server_deployment.docker_login import login_with_server_docker_mode
from fedml.cli.server_deployment.docker_login import logout_with_server_docker_mode
from fedml.cli.server_deployment.docker_login import logs_with_server_docker_mode
from fedml.cli.edge_deployment.client_diagnosis import ClientDiagnosis
from fedml.cli.comm_utils import sys_utils
from fedml.cli.model_deployment import device_login_entry
from fedml.cli.model_deployment.device_model_cards import FedMLModelCards


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


@cli.command("diagnosis", help="Diagnosis for open.fedml.ai, AWS S3 service and MQTT service")
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

    "--mqtt_daemon", "-d", default=None, is_flag=True, help="check the connection to mqtt.fedml.ai (1883) with loop mode.",
)
@click.option(
    "--mqtt_s3_backend_server", "-msbs", default=None, is_flag=True, help="check the connection to mqtt.fedml.ai (1883) as mqtt+s3 server.",
)
@click.option(
    "--mqtt_s3_backend_client", "-msbc", default=None, is_flag=True, help="check the connection to mqtt.fedml.ai (1883) as mqtt+s3 client.",
)
@click.option(
    "--mqtt_s3_backend_run_id", "-rid", type=str, default="fedml_diag_9988", help="mqtt+s3 run id.",
)
def mlops_diagnosis(open, s3, mqtt, mqtt_daemon, mqtt_s3_backend_server, mqtt_s3_backend_client, mqtt_s3_backend_run_id):
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
        server_diagnosis_cmd = os.path.join(pip_source_dir, "edge_deployment", "client_diagnosis.py")
        backend_server_process = subprocess.Popen([
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
        client_diagnosis_cmd = os.path.join(pip_source_dir, "edge_deployment", "client_diagnosis.py")
        backend_client_process = subprocess.Popen([
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


@device.command("login", help="Login as model device agent(MDA) on the ModelOps platform (model.fedml.ai).")
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


@device.command("logout", help="Logout from the ModelOps platform (model.fedml.ai)")
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


@model.command("create", help="Create local model repository.")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def create_model(name):
    if FedMLModelCards.get_instance().create_model(name):
        click.echo("Create model {} successfully.".format(name))
    else:
        click.echo("Failed to create model {}.".format(name))


@model.command("delete", help="Delete local model repository.")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def delete_model(name):
    if FedMLModelCards.get_instance().delete_model(name):
        click.echo("Delete model {} successfully.".format(name))
    else:
        click.echo("Failed to delete model {}.".format(name))


@model.command("add", help="Add file to local model repository.")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--path", "-p", type=str, help="path for specific model.",
)
def add_model_files(name, meta, path):
    if FedMLModelCards.get_instance().add_model_files(name, path):
        click.echo("Add file to model {} successfully.".format(name))
    else:
        click.echo("Failed to add file to model {}.".format(name))


@model.command("remove", help="Remove file from local model repository.")
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
def list_remote_models(name, user, api_key, version):
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    FedMLModelCards.get_instance().set_config_version(version)
    model_query_result = FedMLModelCards.get_instance().list_models(name, user, api_key)
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


@model.command("push", help="Push local model repository to ModelOps(model.fedml.ai).")
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
def push_model(name, model_storage_url, model_net_url, user, api_key, version):
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    FedMLModelCards.get_instance().set_config_version(version)
    model_is_from_open = True if model_storage_url is not None and model_storage_url != "" else False
    model_storage_url, model_zip = FedMLModelCards.get_instance().push_model(name, user, api_key,
                                                                             model_storage_url=model_storage_url,
                                                                             model_net_url=model_net_url)
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
def pull_model(name, user, api_key, version):
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    FedMLModelCards.get_instance().set_config_version(version)
    if FedMLModelCards.get_instance().pull_model(name, user, api_key):
        click.echo("Pull model {} successfully.".format(name))
    else:
        click.echo("Failed to pull model {}.".format(name))


@model.command("deploy",
               help="Deploy specific model to ModelOps platform(model.fedml.ai) or just for local debugging deployment.")
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
    "--use_local_deployment", "-ld", default=None, is_flag=True,
    help="deploy local model repository by sending MQTT message(just use for debugging).",
)
def deploy_model(name, on_premise, cloud, devices, user, api_key, params, version, use_local_deployment):
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return

    is_cloud = cloud
    is_on_premise = on_premise
    if cloud is None and on_premise is None:
        is_on_premise = True
    if is_cloud and is_on_premise:
        is_cloud = False

    if is_on_premise:
        device_type = "md.on_premise_device"
    else:
        device_type = "md.fedml_cloud_device"
    FedMLModelCards.get_instance().set_config_version(version)
    if FedMLModelCards.get_instance().deploy_model(name, device_type, devices, user, api_key,
                                                   params, use_local_deployment):
        click.echo("Deploy model {} successfully.".format(name))
    else:
        click.echo("Failed to deploy model {}.".format(name))


@model.group("inference")
def inference():
    """
    Inference models.
    """
    pass


@inference.command("query", help="Query inference parameters for specific model from ModelOps platform(model.fedml.ai).")
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


@inference.command("run", help="Run inference action for specific model from ModelOps platform(model.fedml.ai).")
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
