import os
import shutil

import click

from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager

FEDML_MLOPS_BUILD_PRE_IGNORE_LIST = 'dist-packages,client-package.zip,server-package.zip,__pycache__,*.pyc,*.git'



def login(api_key, version):
    return FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)


def match_resources(yaml_file, prompt):
    return FedMLLaunchManager.get_instance().api_match_resources(yaml_file, prompt)


def authenticate(api_key, version):
    error_code, _ = login(api_key, version)

    # Exit if not able to authenticate successfully
    if error_code:
        exit(0)

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

