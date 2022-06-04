import os
import signal
import subprocess
from os import listdir
from os.path import expanduser

import click
import shutil

import fedml
import psutil
import yaml
from fedml.cli.edge_deployment.yaml_utils import load_yaml_config
from fedml.cli.edge_deployment.client_login import logout as client_logout
from fedml.cli.edge_deployment.client_login import LOCAL_HOME_RUNNER_DIR_NAME as CLIENT_RUNNER_HOME_DIR
from fedml.cli.edge_deployment.client_login import LOCAL_RUNNER_INFO_DIR_NAME as CLIENT_RUNNER_INFO_DIR
from fedml.cli.server_deployment.server_login import logout as server_logout
from fedml.cli.server_deployment.server_login import LOCAL_HOME_RUNNER_DIR_NAME as SERVER_RUNNER_HOME_DIR
from fedml.cli.server_deployment.server_login import LOCAL_RUNNER_INFO_DIR_NAME as SERVER_RUNNER_INFO_DIR


@click.group()
def cli():
    pass


@cli.command("version", help="Display fedml version.")
def mlops_version():
    click.echo("fedml version: " + str(fedml.__version__))


@cli.command("logs", help="Display fedml logs.")
@click.option(
    "--client",
    "-c",
    default=None, is_flag=False,
    help="Display client logs.",
)
@click.option(
    "--server",
    "-s",
    default=None, is_flag=False,
    help="Display server logs.",
)
def mlops_logs(client, server):
    is_client = client
    is_server = server
    if client is False and server is False:
        is_client = True

    if is_client:
        display_client_logs()

    if is_server:
        display_server_logs()


def get_running_info(cs_home_dir, cs_info_dir):
    home_dir = expanduser("~")
    runner_info_file = os.path.join(home_dir, cs_home_dir, "fedml", "data", cs_info_dir, "runner_infos.yaml")
    if os.path.exists(runner_info_file):
        running_info = load_yaml_config(runner_info_file)
        return running_info["run_id"], running_info["edge_id"]
    return 0, 0


def display_client_logs():
    run_id, edge_id = get_running_info(CLIENT_RUNNER_HOME_DIR, CLIENT_RUNNER_INFO_DIR)
    home_dir = expanduser("~")
    log_file = "{}/{}/fedml/logs/fedavg-cross-silo-run-{}-edge-{}.log".format(home_dir,
                                                                              CLIENT_RUNNER_HOME_DIR,
                                                                              str(run_id),
                                                                              str(edge_id))
    if os.path.exists(log_file):
        with open(log_file) as file_handle:
            log_lines = file_handle.readlines()
        for log_line in log_lines:
            click.echo(log_line)


def display_server_logs():
    run_id, edge_id = get_running_info(SERVER_RUNNER_HOME_DIR, SERVER_RUNNER_INFO_DIR)
    home_dir = expanduser("~")
    log_file = "{}/{}/fedml/logs/fedavg-cross-silo-run-{}-edge-{}.log".format(home_dir,
                                                                              SERVER_RUNNER_HOME_DIR,
                                                                              str(run_id),
                                                                              str(edge_id))
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
    "--client",
    "-c",
    default=None, is_flag=False,
    help="login as the FedML client.",
)
@click.option(
    "--server",
    "-s",
    default=None, is_flag=False,
    help="login as the FedML server.",
)
def mlops_login(userid, version, client, server):
    account_id = userid[0]
    platform_url = "open.fedml.ai"
    if version != "release":
        platform_url = "open-{}.fedml.ai".format(version)

    if userid == "":
        click.echo(
            "Please provide your account id in the MLOps platform ({}).".format(platform_url)
        )
        return

    is_client = client
    is_server = server
    if client is False and server is False:
        is_client = True

    if is_client:
        pip_source_dir = os.path.dirname(__file__)
        login_cmd = os.path.join(pip_source_dir, "edge_deployment", "client_login.py")
        click.echo(login_cmd)
        client_logout()
        cleanup_login_process(CLIENT_RUNNER_HOME_DIR, CLIENT_RUNNER_INFO_DIR)
        cleanup_all_fedml_processes("client_login.py", exclude_login=True)
        login_pid = subprocess.Popen(
            [get_python_program(), login_cmd, "-t", "login", "-u", str(account_id), "-v", version]).pid
        save_login_process(CLIENT_RUNNER_HOME_DIR, CLIENT_RUNNER_INFO_DIR, login_pid)

    if is_server:
        pip_source_dir = os.path.dirname(__file__)
        login_cmd = os.path.join(pip_source_dir, "server_deployment", "server_login.py")
        click.echo(login_cmd)
        server_logout()
        cleanup_login_process(SERVER_RUNNER_HOME_DIR, SERVER_RUNNER_INFO_DIR)
        cleanup_all_fedml_processes("server_login.py", exclude_login=True)
        login_pid = subprocess.Popen(
            [get_python_program(), login_cmd, "-t", "login", "-u", str(account_id), "-v", version]).pid
        save_login_process(SERVER_RUNNER_HOME_DIR, SERVER_RUNNER_INFO_DIR, login_pid)


def generate_yaml_doc(yaml_object, yaml_file):
    try:
        file = open(yaml_file, 'w', encoding='utf-8')
        yaml.dump(yaml_object, file)
        file.close()
    except Exception as e:
        pass


def get_python_program():
    python_program = 'python'
    python_version_str = os.popen("python --version").read()
    if python_version_str.find("Python 3.") == -1:
        python_version_str = os.popen("python3 --version").read()
        if python_version_str.find("Python 3.") != -1:
            python_program = 'python3'

    return python_program


def cleanup_login_process(runner_home_dir, runner_info_dir):
    try:
        home_dir = expanduser("~")
        local_pkg_data_dir = os.path.join(home_dir, runner_home_dir, "fedml", "data")
        edge_process_id_file = os.path.join(local_pkg_data_dir, runner_info_dir, "runner-process.id")
        edge_process_info = load_yaml_config(edge_process_id_file)
        edge_process_id = edge_process_info.get('process_id', None)
        if edge_process_id is not None:
            edge_process = psutil.Process(edge_process_id)
            if edge_process is not None:
                os.killpg(os.getpgid(edge_process.pid), signal.SIGTERM)
                # edge_process.terminate()
                # edge_process.join()
        yaml_object = {}
        yaml_object['process_id'] = -1
        generate_yaml_doc(yaml_object, edge_process_id_file)

    except Exception as e:
        pass


def save_login_process(runner_home_dir, runner_info_dir, edge_process_id):
    home_dir = expanduser("~")
    local_pkg_data_dir = os.path.join(home_dir, runner_home_dir, "fedml", "data")
    try:
        os.makedirs(local_pkg_data_dir)
    except Exception as e:
        pass
    try:
        os.makedirs(os.path.join(local_pkg_data_dir, runner_info_dir))
    except Exception as e:
        pass

    try:
        edge_process_id_file = os.path.join(local_pkg_data_dir, runner_info_dir, "runner-process.id")
        yaml_object = {}
        yaml_object['process_id'] = edge_process_id
        generate_yaml_doc(yaml_object, edge_process_id_file)
    except Exception as e:
        pass


def cleanup_all_fedml_processes(login_program, exclude_login=False):
    # Cleanup all fedml relative processes.
    for process in psutil.process_iter():
        try:
            pinfo = process.as_dict(attrs=['pid', 'name', "cmdline"])
            for cmd in pinfo["cmdline"]:
                if exclude_login:
                    if str(cmd).find("fedml_config.yaml") != -1:
                        click.echo("find fedml process at {}.".format(process.pid))
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        # process.terminate()
                        # process.join()
                else:
                    if str(cmd).find(login_program) != -1 or str(cmd).find("fedml_config.yaml") != -1:
                        click.echo("find fedml process at {}.".format(process.pid))
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        # process.terminate()
                        # process.join()
        except Exception as e:
            pass


@cli.command(
    "logout", help="Logout from MLOps platform (open.fedml.ai)"
)
@click.option(
    "--client",
    "-c",
    default=None, is_flag=False,
    help="logout from the FedML client.",
)
@click.option(
    "--server",
    "-s",
    default=None, is_flag=False,
    help="logout from the FedML server.",
)
def mlops_logout(client, server):
    is_client = client
    is_server = server
    if client is False and server is False:
        is_client = True

    if is_client:
        client_logout()
        cleanup_login_process(CLIENT_RUNNER_HOME_DIR, CLIENT_RUNNER_INFO_DIR)
        cleanup_all_fedml_processes("client_login.py")
    if is_server:
        server_logout()
        cleanup_login_process(CLIENT_RUNNER_HOME_DIR, CLIENT_RUNNER_INFO_DIR)
        cleanup_all_fedml_processes("server_login.py")


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
def mlops_build(type, source_folder, entry_point, config_folder, dest_folder):
    click.echo("Argument for type: " + type)
    click.echo("Argument for source folder: " + source_folder)
    click.echo("Argument for entry point: " + entry_point)
    click.echo("Argument for config folder: " + config_folder)
    click.echo("Argument for destination package folder: " + dest_folder)

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

    pip_source_dir = os.path.dirname(__file__)
    pip_build_path = os.path.join(pip_source_dir, "build-package")
    shutil.copytree(pip_build_path, mlops_build_path)

    if type == "client":
        result = build_mlops_package(
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

    shutil.rmtree(mlops_dest_conf, ignore_errors=True)
    shutil.rmtree(mlops_dest, ignore_errors=True)
    try:
        shutil.copytree(mlops_src, mlops_dest, copy_function=shutil.copy)
    except Exception as e:
        pass
    try:
        shutil.copytree(mlops_conf, mlops_dest_conf, copy_function=shutil.copy)
    except Exception as e:
        pass
    try:
        os.remove(os.path.join(mlops_dest_conf, "mqtt_config.yaml"))
        os.remove(os.path.join(mlops_dest_conf, "s3_config.yaml"))
    except Exception as e:
        pass

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

    mlops_pkg_conf_file = open(mlops_pkg_conf, mode="w")
    mlops_pkg_conf_file.writelines(
        [
            "entry_config: \n",
            "  entry_file: " + mlops_dest_entry + "\n",
            "  conf_file: " + os.path.join("config", "fedml_config.yaml") + "\n",
            "dynamic_args:\n",
            "  rank: " + rank + "\n",
            "  run_id: ${FEDSYS.RUN_ID}\n",
            # "  data_cache_dir: ${FEDSYS.PRIVATE_LOCAL_DATA}\n",
            "  data_cache_dir: /fedml/fedml-package/fedml/data\n",
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

    shutil.rmtree(mlops_build_path, ignore_errors=True)

    return 0


# TODO:
# Maybe not be supported as follows
# fedml mlops --action deploy
# fedml mlops --action logout
# fedml mlops --action on
# fedml mlops --action off
# fedml mlops --action clean
# fedml mlops --action exit
# So we should do as follows
# fedml deploy
# fedml logout
# fedml on
# fedml off
# fedml clean
# fedml exit


if __name__ == "__main__":
    cli()
