import os
import json
import shutil
from os.path import expanduser

import click

import fedml
from fedml.computing.scheduler.comm_utils.constants import SchedulerConstants
from fedml.computing.scheduler.scheduler_entry.job_manager import FedMLJobManager

from fedml.computing.scheduler.slave.client_constants import ClientConstants

from fedml.computing.scheduler.env.collect_env import collect_env

from fedml.computing.scheduler.slave.client_diagnosis import ClientDiagnosis
from fedml.computing.scheduler.comm_utils import sys_utils
from fedml.computing.scheduler.model_scheduler.device_model_cards import FedMLModelCards
from fedml.computing.scheduler.comm_utils.platform_utils import platform_is_valid
from fedml.computing.scheduler.scheduler_entry.launch_manager import FedMLLaunchManager
from fedml.computing.scheduler.slave.docker_login import logs_with_docker_mode
from fedml.computing.scheduler.master.docker_login import logs_with_server_docker_mode
from fedml.cli import cli_utils

from prettytable import PrettyTable


@click.group()
@click.help_option("--help", "-h")
def cli():
    pass


@cli.command("version", help="Display fedml version.")
@click.help_option("--help", "-h")
def fedml_version():
    click.echo("fedml version: " + str(fedml.__version__))


@cli.command("status", help="Display fedml client training status.")
@click.help_option("--help", "-h")
def fedml_status():
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
    help="show resource type at which version of FedML® Launch platform. It should be dev, test or release",
)
def fedml_launch_show_resource_type(version):
    FedMLLaunchManager.get_instance().set_config_version(version)
    resource_type_list = FedMLLaunchManager.get_instance().show_resource_type()
    if resource_type_list is not None and len(resource_type_list) > 0:
        click.echo("All available resource type is as follows.")
        resource_table = PrettyTable(['Resource Type', 'GPU Type'])
        for type_item in resource_type_list:
            resource_table.add_row([type_item[0], type_item[1]])
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
def fedml_logs(client, server, docker, docker_rank):
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
        cli_utils.display_client_logs()

    if is_server:
        if is_docker:
            logs_with_server_docker_mode(docker_rank)
            return
        cli_utils.display_server_logs()


@cli.command("login", help="Bind to the FedML® Launch platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.argument("userid", nargs=-1)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="bind to which version of FedML® Launch platform. It should be dev, test or release",
)
@click.option(
    "--client", "-c", default=None, is_flag=True, help="bind as the FedML client.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="bind as the FedML server.",
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
    default="",
    help="run as the role (options: client, edge_simulator, gpu_supplier, "
         "edge_server, cloud_agent, cloud_server, gpu_master_server.",
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
    "--docker", "-d", default=None, is_flag=True, help="bind with docker mode at the client agent.",
)
@click.option(
    "--docker-rank", "-dr", default="1", help="docker client rank index (from 1 to n).",
)
def fedml_login(userid, version, client, server,
                api_key, local_server, role, runner_cmd, device_id, os_name,
                docker, docker_rank):
    cli_utils.fedml_device_bind_wrapper(
        userid, version, client, server,
        api_key, local_server, role, runner_cmd, device_id, os_name,
        docker, docker_rank
    )


@cli.command("logout", help="unbind from the FedML® Launch platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.option(
    "--client", "-c", default=None, is_flag=True, help="unbind from the FedML client.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="unbind from the FedML server.",
)
@click.option(
    "--docker", "-d", default=None, is_flag=True, help="unbind from docker mode at the client agent.",
)
@click.option(
    "--docker-rank", "-dr", default=None, help="docker client rank index (from 1 to n).",
)
def fedml_logout(client, server, docker, docker_rank):
    cli_utils.fedml_device_unbind_wrapper(client, server, docker, docker_rank)


@cli.group("device")
@click.help_option("--help", "-h")
def fedml_device():
    """
    Manage devices on the FedML® Launch platform (open.fedml.ai).
    """
    pass


@fedml_device.command("bind", help="Bind to the FedML® Launch platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.argument("userid", nargs=-1)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="bind to which version of FedML® Launch platform. It should be dev, test or release",
)
@click.option(
    "--client", "-c", default=None, is_flag=True, help="bind as the FedML client.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="bind as the FedML server.",
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
    default="",
    help="run as the role (options: client, edge_simulator, gpu_supplier, "
         "edge_server, cloud_agent, cloud_server, gpu_master_server.",
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
    "--docker", "-d", default=None, is_flag=True, help="bind with docker mode at the client agent.",
)
@click.option(
    "--docker-rank", "-dr", default="1", help="docker client rank index (from 1 to n).",
)
def fedml_device_bind(
        userid, version, client, server,
        api_key, local_server, role, runner_cmd, device_id, os_name,
        docker, docker_rank
):
    cli_utils.fedml_device_bind_wrapper(
        userid, version, client, server,
        api_key, local_server, role, runner_cmd, device_id, os_name,
        docker, docker_rank
    )


@fedml_device.command("unbind", help="unbind from the FedML® Launch platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.option(
    "--client", "-c", default=None, is_flag=True, help="unbind from the FedML client.",
)
@click.option(
    "--server", "-s", default=None, is_flag=True, help="unbind from the FedML server.",
)
@click.option(
    "--docker", "-d", default=None, is_flag=True, help="unbind from docker mode at the client agent.",
)
@click.option(
    "--docker-rank", "-dr", default=None, help="docker client rank index (from 1 to n).",
)
def fedml_device_unbind(client, server, docker, docker_rank):
    cli_utils.fedml_device_unbind_wrapper(client, server, docker, docker_rank)


@cli.command("build", help="Build packages for the FedML® Launch platform (open.fedml.ai)")
@click.help_option("--help", "-h")
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="octopus",
    help="The platform name at the FedML® Launch platform (options: octopus, parrot, spider, beehive, falcon, launch).",
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
def fedml_build(platform, type, source_folder, entry_point, config_folder, dest_folder, ignore):
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
            "Now, you are building the fedml packages which will be used in the FedML® Launch platform "
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
            "Then you may upload the packages on the configuration page in the FedML® Launch platform to start the "
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

    ignore_list = "{},{}".format(ignore, cli_utils.FEDML_MLOPS_BUILD_PRE_IGNORE_LIST)
    pip_source_dir = os.path.dirname(__file__)
    pip_source_dir = os.path.dirname(pip_source_dir)
    pip_build_path = os.path.join(pip_source_dir, "computing", "scheduler", "build-package")
    build_dir_ignore = "__pycache__,*.pyc,*.git"
    build_dir_ignore_list = tuple(build_dir_ignore.split(','))
    shutil.copytree(pip_build_path, mlops_build_path,
                    ignore_dangling_symlinks=True, ignore=shutil.ignore_patterns(*build_dir_ignore_list))

    if type == "client":
        result = cli_utils.build_mlops_package(
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
        result = cli_utils.build_mlops_package(
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
def fedml_diagnosis(open, s3, mqtt, mqtt_daemon, mqtt_s3_backend_server, mqtt_s3_backend_client,
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
def fedml_env():
    collect_env()


@cli.group("launch", cls=cli_utils.DefaultCommandGroup, default_command='run')
@click.help_option("--help", "-h")
def fedml_launch():
    """
    Manage resources on the FedML® Launch platform (open.fedml.ai).
    """
    pass


@fedml_launch.command("cancel", help="Cancel job at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("job_id", nargs=-1)
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the FedML® Launch platform (options: octopus, parrot, spider, beehive, falcon, launch, "
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
def fedml_launch_cancel(job_id, platform, api_key, version):
    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    api_key = FedMLLaunchManager.get_api_key()
    cli_utils.stop_job_wrapper(platform, job_id[0], api_key, version)


@fedml_launch.command("log", help="View the job list at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("job_id", nargs=-1)
@click.option(
    "--platform",
    "-pf",
    type=str,
    default="falcon",
    help="The platform name at the FedML® Launch platform (options: octopus, parrot, spider, beehive, falcon, launch, "
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
def fedml_launch_log(job_id, platform, api_key, version):
    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    FedMLLaunchManager.get_instance().api_launch_log(job_id[0], 0, 0, need_all_logs=True)


@fedml_launch.command("queue", help="View the job queue at the FedML® Launch platform (open.fedml.ai)", )
@click.help_option("--help", "-h")
@click.argument("group_id", nargs=-1)
def fedml_launch_queue(group_id):
    pass


@fedml_launch.command(
    "run", help="Launch job at the FedML® Launch platform (open.fedml.ai)",
    context_settings={"ignore_unknown_options": True}
)
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
    help="launch job to which version of FedML® Launch platform. It should be dev, test or release",
)
def fedml_launch_run(yaml_file, api_key, group, version):
    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    FedMLLaunchManager.get_instance().set_config_version(version)
    FedMLLaunchManager.get_instance().api_launch_job(yaml_file[0], None)


@fedml_launch.group("utils")
@click.help_option("--help", "-h")
def fedml_launch_utils():
    """
    Manage launch related utils on the MLOps platform.
    """
    pass


@fedml_launch_utils.command(
    "launch-octopus", help="Launch job at the FedML® Launch platform (open.fedml.ai)",
    context_settings={"ignore_unknown_options": True}
)
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
    "--devices_server", "-ds", type=str, default="",
    help="The server to run the launching job, for the launch platform, we do not need to set this option."
)
@click.option(
    "--devices_edges", "-de", type=str, default="",
    help="The edge devices to run the launching job. Seperated with ',', e.g. 705,704. "
         "for the launch platform, we do not need to set this option."
)
@click.option(
    "--version",
    "-v",
    type=str,
    default="release",
    help="launch job to which version of MLOps platform. It should be dev, test or release",
)
def fedml_launch_utils_launch_octopus(yaml_file, api_key, group, devices_server, devices_edges, version):
    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    FedMLLaunchManager.get_instance().set_config_version(version)
    FedMLLaunchManager.get_instance().platform_type = SchedulerConstants.PLATFORM_TYPE_OCTOPUS
    FedMLLaunchManager.get_instance().device_server = devices_server
    FedMLLaunchManager.get_instance().device_edges = devices_edges
    FedMLLaunchManager.get_instance().api_launch_job(yaml_file[0], None)


@fedml_launch_utils.command("start-job", help="Start a job at the MLOps platform.")
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
def fedml_launch_utils_start_job(platform, project_name, application_name, job_name, devices_server, devices_edges, user, api_key,
              version):
    if not platform_is_valid(platform):
        return

    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

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


@fedml_launch_utils.command("list-job", help="List jobs from the MLOps platform.")
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
    help="list jobs at which version of MLOps platform. It should be dev, test or release",
)
def fedml_launch_utils_list_jobs(platform, job_id, api_key, version):
    if not platform_is_valid(platform):
        return

    error_code, _ = FedMLLaunchManager.get_instance().fedml_login(api_key=api_key, version=version)
    if error_code != 0:
        click.echo("Please check if your API key is valid.")
        return

    FedMLLaunchManager.get_instance().set_config_version(version)
    FedMLLaunchManager.get_instance().list_jobs(job_id)


@fedml_launch_utils.command("stop-job", help="Stop a job from the MLOps platform.")
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
def fedml_launch_utils_stop_job(platform, job_id, api_key, version):
    cli_utils.stop_job_wrapper(platform, job_id, api_key, version)


@cli.group("model")
@click.help_option("--help", "-h")
def fedml_model():
    """
    Deploy and infer models.
    """
    pass


@fedml_model.command("create", help="Create local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_create(name):
    if FedMLModelCards.get_instance().create_model(name):
        click.echo("Create model {} successfully.".format(name))
    else:
        click.echo("Failed to create model {}.".format(name))


@fedml_model.command("delete", help="Delete local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_delete(name):
    if FedMLModelCards.get_instance().delete_model(name):
        click.echo("Delete model {} successfully.".format(name))
    else:
        click.echo("Failed to delete model {}.".format(name))


@fedml_model.command("add", help="Add file to local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--path", "-p", type=str, help="path for specific model.",
)
def fedml_model_add(name, path):
    if FedMLModelCards.get_instance().add_model_files(name, path):
        click.echo("Add file to model {} successfully.".format(name))
    else:
        click.echo("Failed to add file to model {}.".format(name))


@fedml_model.command("remove", help="Remove file from local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--file", "-f", type=str, help="file name for specific model.",
)
def fedml_model_remove(name, file):
    if FedMLModelCards.get_instance().remove_model_files(name, file):
        click.echo("Remove file from model {} successfully.".format(name))
    else:
        click.echo("Failed to remove file from model {}.".format(name))


@fedml_model.command("list", help="List model in the local model repository.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_list(name):
    models = FedMLModelCards.get_instance().list_models(name)
    if len(models) <= 0:
        click.echo("Model list is empty.")
    else:
        for model_item in models:
            click.echo(model_item)
        click.echo("List model {} successfully.".format(name))


@fedml_model.command("list-remote", help="List models in the remote model repository.")
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
def fedml_model_list_remote(name, user, api_key, version, local_server):
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


@fedml_model.command("package", help="Build local model repository as zip model package.")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_package(name):
    model_zip = FedMLModelCards.get_instance().build_model(name)
    if model_zip != "":
        click.echo("Build model package {} successfully".format(name))
        click.echo("The local package is located at {}.".format(model_zip))
    else:
        click.echo("Failed to build model {}.".format(name))


@fedml_model.command("push", help="Push local model repository to ModelOps(open.fedml.ai).")
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
def fedml_model_push(name, model_storage_url, model_net_url, user, api_key, version, local_server):
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


@fedml_model.command("pull", help="Pull remote model(ModelOps) to local model repository.")
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
def fedml_model_pull(name, user, api_key, version, local_server):
    if user is None or api_key is None:
        click.echo("You must provide arguments for User Id and Api Key (use -u and -k options).")
        return
    FedMLModelCards.get_instance().set_config_version(version)
    if FedMLModelCards.get_instance().pull_model(name, user, api_key, local_server):
        click.echo("Pull model {} successfully.".format(name))
    else:
        click.echo("Failed to pull model {}.".format(name))


@fedml_model.command("deploy",
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
def fedml_model_deploy(name, on_premise, cloud, devices, user, api_key, params, version,
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


@fedml_model.group("inference")
def fedml_model_inference():
    """
    Inference models.
    """
    pass


@fedml_model_inference.command("query",
                               help="Query inference parameters for specific model from ModelOps platform(open.fedml.ai).")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
def fedml_model_inference_query(name):
    inference_output_url, model_metadata, model_config = FedMLModelCards.get_instance().query_model(name)
    if inference_output_url != "":
        click.echo("Query model {} successfully.".format(name))
        click.echo("infer url: {}.".format(inference_output_url))
        click.echo("model metadata: {}.".format(model_metadata))
        click.echo("model config: {}.".format(model_config))
    else:
        click.echo("Failed to query model {}.".format(name))


@fedml_model_inference.command("run",
                               help="Run inference action for specific model from ModelOps platform(open.fedml.ai).")
@click.help_option("--help", "-h")
@click.option(
    "--name", "-n", type=str, help="model name.",
)
@click.option(
    "--data", "-d", type=str, help="input data for model inference.",
)
def fedml_model_inference_run(name, data):
    infer_out_json = FedMLModelCards.get_instance().inference_model(name, data)
    if infer_out_json != "":
        click.echo("Inference model {} successfully.".format(name))
        click.echo("Result: {}.".format(infer_out_json))
    else:
        click.echo("Failed to inference model {}.".format(name))


if __name__ == "__main__":
    cli()
