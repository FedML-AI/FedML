import os
import shutil
from os.path import expanduser

import click

from fedml.api.modules.utils import build_mlops_package
from fedml.computing.scheduler.comm_utils.platform_utils import validate_platform
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.scheduler_entry.constants import Constants


def build(platform, type, source_folder, entry_point, config_folder, dest_folder, ignore, package_type="default"):
    click.echo("Argument for type: " + type)
    click.echo("Argument for source folder: " + source_folder)
    click.echo("Argument for entry point: " + entry_point)
    click.echo("Argument for config folder: " + config_folder)
    click.echo("Argument for destination package folder: " + dest_folder)
    click.echo("Argument for ignore lists: " + ignore)

    validate_platform(platform)

    if type == "client" or type == "server":
        click.echo(
            "Now, you are building the fedml packages which will be used in the FedML® Nexus AI Platform "
            "platform."
        )
        click.echo(
            "The packages will be used for your training."
        )
        click.echo(
            "When the building process is completed, you will find the packages in the directory as follows: "
            + os.path.join(dest_folder, "dist-packages")
            + "."
        )
        click.echo(
            "Then you may upload the packages on the configuration page in the FedML® Nexus AI Platform to "
            "start your training flow."
        )
        click.echo("Building...")
    else:
        click.echo("You should specify the type argument value as client or server.")
        exit(-1)

    home_dir = expanduser("~")
    fedml_dir = os.path.join(home_dir, ".fedml")
    os.makedirs(fedml_dir, exist_ok=True)
    mlops_build_path = os.path.join(fedml_dir, "fedml-mlops-build")
    try:
        shutil.rmtree(mlops_build_path, ignore_errors=True)
    except Exception as e:
        pass

    # Read the gitignore file
    gitignore_file = os.path.join(source_folder, ".gitignore")
    if os.path.exists(gitignore_file):
        ignore_list_str = sys_utils.read_gitignore_file(gitignore_file)
        ignore = f"{ignore},{ignore_list_str}"

    ignore_list = "{},{}".format(ignore, Constants.FEDML_MLOPS_BUILD_PRE_IGNORE_LIST)
    pip_source_dir = os.path.dirname(__file__)
    pip_source_dir = os.path.dirname(pip_source_dir)
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
            package_type=package_type
        )
        if result != 0:
            exit(result)
        click.echo("You have finished all building process. ")
        click.echo(
            "Now you may use "
            + os.path.join(dest_folder, "dist-packages", "client-package.zip")
            + " to start your training"
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
            package_type=package_type
        )
        if result != 0:
            exit(result)

        click.echo("You have finished all building process. ")
        click.echo(
            "Now you may use "
            + os.path.join(dest_folder, "dist-packages", "server-package.zip")
            + " to start your training."
        )
