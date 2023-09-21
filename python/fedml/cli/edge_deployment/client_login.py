
import argparse
import os

from fedml.computing.scheduler.slave.client_login import login, logout

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
    parser.add_argument("--device_id", "-id", type=str, default="0")
    parser.add_argument("--os_name", "-os", type=str, default="")
    parser.add_argument("--api_key", "-k", type=str, default="")
    parser.add_argument("--no_gpu_check", "-ngc", type=int, default=1)
    args = parser.parse_args()
    
    args.user = args.user
    if args.type == 'login':
        login(args)
    else:
        logout()


