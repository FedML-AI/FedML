import os
import shutil
from os.path import expanduser

import click
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.comm_utils.platform_utils import platform_is_valid
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.scheduler_entry.job_manager import FedMLJobManager
from fedml.computing.scheduler.slave.client_constants import ClientConstants

FEDML_MLOPS_BUILD_PRE_IGNORE_LIST = 'dist-packages,client-package.zip,server-package.zip,__pycache__,*.pyc,*.git'


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


def stop_job_wrapper(platform, job_id, api_key, version, show_hint_texts=True):
    if not platform_is_valid(platform):
        return

    FedMLJobManager.get_instance().set_config_version(version)
    is_stopped = FedMLJobManager.get_instance().stop_job(platform, job_id, api_key)
    if show_hint_texts:
        if is_stopped:
            click.echo("Job has been stopped.")
        else:
            click.echo("Failed to stop the job, please check your network connection "
                       "and make sure be able to access the FedML® Launch platform.")
    return is_stopped


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