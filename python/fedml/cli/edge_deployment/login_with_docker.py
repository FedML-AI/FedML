import os
import platform
import uuid

import click
from os.path import expanduser


def login_with_docker_mode(userid, version):
    account_id = userid

    # Get os name
    sys_name = platform.system()
    if sys_name == "Windows":
        click.echo("Login into the FedML MLOps platform on the Windows platform will be coming soon. Please stay tuned.")
        return
    if sys_name == "Darwin":
        sys_name = "MacOS"

    # Get home directory
    cur_dir = expanduser("~")
    cur_dir = os.path.join(cur_dir, "fedml-client")

    # Set default version if the version argument is empty
    if version == "":
        version = "release"

    # Set default registry server to AWS
    fedml_is_using_aws_pubic_ecr = 1
    if version != "release":
        fedml_is_using_aws_pubic_ecr = 0

    # Set registry server based on the version.
    if fedml_is_using_aws_pubic_ecr == 1:
        registry_server = "public.ecr.aws"
        image_dir = "/x6k8q1x9"
        client_registry_server = registry_server
        client_image_dir = image_dir
    else:
        registry_server = "registry.fedml.ai"
        image_dir = "/fedml-public-server"
        client_registry_server = registry_server
        client_image_dir = image_dir

    # Set image tags based on the version
    tag = "dev"
    client_tag = "dev"
    if version == "local":
        tag = "local"
        client_tag = "local"

    # Echo current development version.
    click.echo("Deployment version: {}".format(version))

    # Set client agent image path and client image path
    image_path = image_dir + "/fedml-client-agent:" + tag
    client_agent_image = registry_server + image_path
    client_base_image = registry_server + image_dir + "/fedml-cross-silo-cpu:" + client_tag

    # Get device id based on your machine MAC address.
    os_name = sys_name
    device_id = hex(uuid.getnode())
    click.echo("OS Name: {}".format(os_name))
    click.echo("current dir: {}".format(cur_dir))

    # Set environment variables for client agent docker
    env_account_id = account_id
    env_config_file = "/fedml/fedml_config/config.yaml"
    env_current_running_dir = cur_dir
    env_client_version = client_tag
    env_current_os_name = os_name
    env_current_device_id = device_id
    env_docker_registry_public_server = client_registry_server
    env_docker_registry_root_dir = client_image_dir

    # Cleanup the running docker
    click.echo("The FedML client agent is being deployed, please wait for a moment...")
    os.system("sudo chmod 777 /var/run/docker.sock")
    os.system("docker stop `docker ps -a |grep fedml_container_run_ |grep _edge_ |awk -F' ' '{print $1}'` >/dev/null 2>&1")
    os.system("docker rm `docker ps -a |grep fedml_container_run_ |grep _edge_ |awk -F' ' '{print $1}'` >/dev/null 2>&1")
    os.system("docker stop `docker ps -a |grep fedml-container-run- |grep _edge_ |awk -F' ' '{print $1}'` >/dev/null 2>&1")
    os.system("docker rm `docker ps -a |grep fedml-container-run- |grep _edge_ |awk -F' ' '{print $1}'` >/dev/null 2>&1")
    os.system("docker stop `docker ps -a |grep '/fedml-server/' |awk -F' ' '{print $1}'` >/dev/null 2>&1")
    os.system("docker rm `docker ps -a |grep '/fedml-server/' |awk -F' ' '{print $1}'`  >/dev/null 2>&1")
    os.system("docker rmi `docker ps -a |grep '/fedml-server/' |awk -F' ' '{print $1}'` >/dev/null 2>&1")

    # Pull client agent docker
    click.echo("Using docker daemon mode")
    click.echo(".........................")
    os.system("docker pull " + client_agent_image)
    os.system("docker stop fedml-client-agent >/dev/null 2>&1")
    os.system("docker rm fedml-client-agent >/dev/null  2>&1")
    os.system("rm -rf " + cur_dir + "/fedml_config >/dev/null  2>&1")

    # Compose the command for running the client agent docker
    docker_run_cmd = "docker run --name fedml-client-agent "
    if os_name == "MacOS":
        docker_run_cmd = "docker run --name fedml-client-agent "
    elif os_name == "Linux":
        docker_run_cmd = "docker run --name fedml-client-agent -v $(which  docker):/usr/bin/docker "
    elif os_name == "Windows":
        docker_run_cmd = "docker run --name fedml-client-agent "

    docker_run_cmd = docker_run_cmd + \
                     " -v /var/run/docker.sock:/var/run/docker.sock" + \
                     " -v " + env_current_running_dir + "/fedml_data:/fedml/data" + \
                     " -v " + env_current_running_dir + "/fedml_data:/fedml/fedml-package/fedml/data" + \
                     " -v " + env_current_running_dir + "/fedml_config:/fedml/conf" + \
                     " -v " + env_current_running_dir + "/fedml_run_state:/fedml/fedml_run_state" + \
                     " --env ACCOUNT_ID=" + env_account_id + \
                     " --env CONFIG_FILE=" + env_config_file + \
                     " --env CURRENT_RUNNING_DIR=" + env_current_running_dir + \
                     " --env CLIENT_VERSION=" + env_client_version + \
                     " --env OS_NAME=" + env_current_os_name + \
                     " --env CURRENT_DEVICE_ID=" + env_current_device_id + \
                     " --env DOCKER_REGISTRY_PUBLIC_SERVER=" + env_docker_registry_public_server + \
                     " --env DOCKER_REGISTRY_ROOT_DIR=" + env_docker_registry_root_dir + \
                     " -d " + client_agent_image + ">/dev/null 2>&1"

    # Run the client agent docker
    os.system(docker_run_cmd)

    # Get the running state for the client agent docker
    deployed = os.popen("docker ps -a |grep fedml-client-agent:" + tag + " |awk -F'Up' '{print $2}'")
    if deployed != "":
        click.echo("Congratulations, you have deployed the FedML client agent successfully!")
        device_id_to_display = ""
        if env_current_device_id != "":
            device_id_to_display = env_current_device_id
        else:
            device_id_to_display = os.popen("cat " + cur_dir + "/fedml_run_state/fedml_client_device_id")

        click.echo("Your device id is " + device_id_to_display + ". You may review the device in the MLOps edge device list.")
        click.echo("--------------------------------------------------------------------------------------------")
        click.echo("Now the system will post-process to pull the FedML client docker image to your localhost.")
        click.echo("You may do other things to start your FedML flow. (User Guide: https://doc.fedml.ai)")
        click.echo("You just need to keep this window not closed until the processing is finished.")
        os.system("docker pull " + client_base_image)
        click.echo("Great, you have succeeded to complete all the running processes.")
    else:
        click.echo("Oops, you failed to deploy the FedML client agent.")
        click.echo("Please check whether your Docker Application is installed and running normally!")


if __name__ == "__main__":
    login_with_docker_mode("105", "test")



