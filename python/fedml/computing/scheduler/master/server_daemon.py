
import argparse
import os
import time
import platform
import logging

import fedml
from fedml.computing.scheduler.comm_utils.sys_utils import cleanup_all_fedml_server_api_processes,\
    cleanup_all_fedml_server_learning_processes,cleanup_all_fedml_server_login_processes, get_python_program, \
    daemon_ota_upgrade
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.computing.scheduler.comm_utils.run_process_utils import RunProcessUtils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to MLOps platform")
    parser.add_argument("--user", "-u", type=str,
                        help='account id at MLOps platform')
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--role", "-r", type=str, default="local")
    parser.add_argument("--runner_cmd", "-rc", type=str, default="{}")
    parser.add_argument("--device_id", "-id", type=str, default="0")
    parser.add_argument("--os_name", "-os", type=str, default="")
    parser.add_argument("--api_key", "-k", type=str, default="")
    parser.add_argument("--local_on_premise_platform_host", "-lp", type=str, default="127.0.0.1")
    parser.add_argument("--local_on_premise_platform_port", "-lpp", type=int, default=80)

    args = parser.parse_args()
    args.user = args.user

    if args.local_on_premise_platform_host != "127.0.0.1":
        fedml.set_local_on_premise_platform_host(args.local_on_premise_platform_host)
    if args.local_on_premise_platform_port != 80:
        fedml.set_local_on_premise_platform_port(args.local_on_premise_platform_port)

    pip_source_dir = os.path.dirname(__file__)
    login_cmd = os.path.join(pip_source_dir, "server_login.py")
    login_exit_file = os.path.join(ServerConstants.get_log_file_dir(), "exited.log")

    try:
        if os.path.exists(login_exit_file):
            os.remove(login_exit_file)
    except Exception as e:
        logging.error(f"Remove failed | Exception: {e}")
        pass

    log_line_count = 0
    retry_count = 0

    while True:
        try:
            ServerConstants.cleanup_run_process(None)
            cleanup_all_fedml_server_api_processes()
            cleanup_all_fedml_server_learning_processes()
            cleanup_all_fedml_server_login_processes("server_login.py", clean_process_group=False)
        except Exception as e:
            logging.error(f"Cleanup failed | Exception: {e}")
            pass

        # daemon_ota_upgrade(args)

        if platform.system() == "Windows" or \
                args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_CLOUD_SERVER_INDEX]:
            login_pid = ServerConstants.exec_console_with_shell_script_list(
                [
                    get_python_program(),
                    "-W",
                    "ignore",
                    login_cmd,
                    "-t",
                    "login",
                    "-u",
                    str(args.user),
                    "-v",
                    args.version,
                    "-r",
                    args.role,
                    "-rc",
                    args.runner_cmd,
                    "-id",
                    args.device_id,
                    "-os",
                    args.os_name,
                    "-k",
                    args.api_key
                ]
            )
            ret_code, exec_out, exec_err = ServerConstants.get_console_sys_out_pipe_err_results(login_pid)
            time.sleep(3)

            if args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_CLOUD_SERVER_INDEX]:
                break
        else:
            login_logs = os.path.join(ServerConstants.get_log_file_dir(), "login.log")
            run_login_cmd = f"nohup {get_python_program()} -W ignore {login_cmd} -t login -u {args.user} " \
                            f"-v {args.version} -r {args.role} -rc {args.runner_cmd} -id {args.device_id} " \
                            f"-k {args.api_key} > {login_logs} 2>&1 &"
            if args.os_name != "":
                run_login_cmd += f" -os {args.os_name}"
            os.system(run_login_cmd)

            login_pids = RunProcessUtils.get_pid_from_cmd_line(login_cmd)
            while len(login_pids) > 0:
                with open(login_logs, "r") as f:
                    log_list = f.readlines()
                    if len(log_list) > log_line_count:
                        print("".join(log_list[log_line_count:len(log_list)]))
                        log_line_count = len(log_list)
                time.sleep(3)
                login_pids = RunProcessUtils.get_pid_from_cmd_line(login_cmd)
                login_exit_file = os.path.join(ServerConstants.get_log_file_dir(), "exited.log")
                retry_flag = False

                if os.path.exists(login_exit_file):
                    print(f"[Server] Login process is exited, check the exit file {login_exit_file}")
                    if retry_count > 3:
                        if args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_CLOUD_AGENT_INDEX]:
                            retry_count = 0
                        else:
                            print(f"Retry count is over 3 times, exit the process. Check the log file for more details. "
                                  f"Login logs: {login_logs}, Exit file: {login_exit_file}")
                            exit(1)
                    retry_flag = True

                if len(login_pids) == 0:
                    message = f"[Server] Login process is exited, check the log file {login_logs}"
                    print(message)
                    if retry_count >= 3:
                        if args.role == ServerConstants.login_role_list[ServerConstants.LOGIN_MODE_CLOUD_AGENT_INDEX]:
                            retry_count = 0
                        else:
                            print(f"Retry count is over 3 times, exit the process. Check the log file for more details. "
                                  f"Login logs: {login_logs}, Exit file: {login_exit_file}")
                            exit(1)
                    retry_flag = True

                if retry_flag:
                    retry_count += 1

            time.sleep(3)
            print(f"[Server] Retry to start the login process. Retry count: {retry_count}")


