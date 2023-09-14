import click

from fedml.computing.scheduler.model_scheduler import device_login_entry


@click.group("device")
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
