import argparse
import os
import fedml
from fedml.computing.scheduler.slave.united_agents import FedMLUnitedAgent


def logout():
    FedMLUnitedAgent.get_instance().logout()


if __name__ == "__main__":
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    os.environ.setdefault('PYTHONWARNINGS', 'ignore:semaphore_tracker:UserWarning')
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to MLOps platform")
    parser.add_argument("--user", "-u", type=str,
                        help='account id at MLOps platform')
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--local_server", "-ls", type=str, default="127.0.0.1")
    parser.add_argument("--role", "-r", type=str, default="client")
    parser.add_argument("--runner_cmd", "-rc", type=str, default="{}")
    parser.add_argument("--device_id", "-id", type=str, default="0")
    parser.add_argument("--os_name", "-os", type=str, default="")
    parser.add_argument("--api_key", "-k", type=str, default="")
    parser.add_argument("--no_gpu_check", "-ngc", type=int, default=1)
    parser.add_argument("--local_on_premise_platform_host", "-lp", type=str, default="127.0.0.1")
    parser.add_argument("--local_on_premise_platform_port", "-lpp", type=int, default=80)

    args = parser.parse_args()
    args.user = args.user
    if args.api_key == "":
        args.api_key = args.user

    if args.local_on_premise_platform_host != "127.0.0.1":
        fedml.set_local_on_premise_platform_host(args.local_on_premise_platform_host)
    if args.local_on_premise_platform_port != 80:
        fedml.set_local_on_premise_platform_port(args.local_on_premise_platform_port)

    fedml.set_env_version(args.version)
    united_agents = FedMLUnitedAgent.get_instance()
    if args.type == 'login':
        united_agents.login(
            args.api_key, api_key=args.api_key, device_id=args.device_id,
            os_name=args.os_name, role=args.role, runner_cmd=args.runner_cmd)
    else:
        united_agents.logout()
