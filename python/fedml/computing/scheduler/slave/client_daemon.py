
import argparse
import os
import time
import platform
import logging

import fedml
from fedml.computing.scheduler.comm_utils.sys_utils import cleanup_all_fedml_client_api_processes, \
    cleanup_all_fedml_client_learning_processes, cleanup_all_fedml_client_login_processes, get_python_program, \
    daemon_ota_upgrade
from fedml.computing.scheduler.slave.client_constants import ClientConstants
from fedml.computing.scheduler.comm_utils.run_process_utils import RunProcessUtils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to MLOps platform")
    parser.add_argument("--user", "-u", type=str,
                        help='account id at MLOps platform')
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--role", "-r", type=str, default="client")
    parser.add_argument("--device_id", "-id", type=str, default="0")
    parser.add_argument("--os_name", "-os", type=str, default="")
    parser.add_argument("--api_key", "-k", type=str, default="")
    parser.add_argument("--no_gpu_check", "-ngc", type=int, default=1)
    parser.add_argument("--local_on_premise_platform_host", "-lp", type=str, default="127.0.0.1")
    parser.add_argument("--local_on_premise_platform_port", "-lpp", type=int, default=80)

    args = parser.parse_args()
    args.user = args.user

    if args.local_on_premise_platform_host != "127.0.0.1":
        fedml.set_local_on_premise_platform_host(args.local_on_premise_platform_host)
    if args.local_on_premise_platform_port != 80:
        fedml.set_local_on_premise_platform_port(args.local_on_premise_platform_port)

    pip_source_dir = os.path.dirname(__file__)
    login_cmd = os.path.join(pip_source_dir, "client_login.py")
    login_exit_file = os.path.join(ClientConstants.get_log_file_dir(), "exited.log")

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
            ClientConstants.cleanup_run_process(None)
            cleanup_all_fedml_client_api_processes()
            cleanup_all_fedml_client_learning_processes()
            cleanup_all_fedml_client_login_processes("client_login.py", clean_process_group=False)
        except Exception as e:
            logging.error(f"Cleanup failed | Exception: {e}")
            pass


        # daemon_ota_upgrade(args)

        if platform.system() == "Windows":
            login_pid = ClientConstants.exec_console_with_shell_script_list(
                [
                    get_python_program(),
                    "-W",
                    "ignore",
                    login_cmd,
                    "-t",
                    "login",
                    "-u",
                    args.user,
                    "-v",
                    args.version,
                    "-r",
                    args.role,
                    "-id",
                    args.device_id,
                    "-os",
                    args.os_name,
                    "-k",
                    args.api_key,
                    "-ngc",
                    str(args.no_gpu_check)
                ]
            )
            ret_code, exec_out, exec_err = ClientConstants.get_console_sys_out_pipe_err_results(login_pid)
            time.sleep(3)
        else:
            login_logs = os.path.join(ClientConstants.get_log_file_dir(), "login.log")
            run_login_cmd = f"nohup {get_python_program()} -W ignore {login_cmd} -t login -u {args.user} " \
                            f"-v {args.version} -r {args.role} -id {args.device_id} " \
                            f"-k {args.api_key} -ngc {str(args.no_gpu_check)} > {login_logs} 2>&1 &"
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
                login_exit_file = os.path.join(ClientConstants.get_log_file_dir(), "exited.log")
                retry_flag = False

                if os.path.exists(login_exit_file):
                    message = f"[Client] Login process is exited, check the exit file {login_exit_file}"
                    print(message)
                    if retry_count > 3:
                        print(f"Retry count is over 3 times, exit the process. Check the log file for more details. "
                              f"Login logs: {login_logs}, Exit file: {login_exit_file}")
                        exit(1)
                    retry_flag = True

                if len(login_pids) == 0:
                    message = f"[Client] Cannot find login pid {login_pids}, check the log file {login_logs}"
                    print(message)
                    if retry_count >= 3:
                        print(f"Retry count is over 3 times, exit the process. Check the log file for more details. "
                              f"Login logs: {login_logs}, Exit file: {login_exit_file}")
                        exit(1)
                    retry_flag = True

                if retry_flag:
                    retry_count += 1

            time.sleep(3)
            print(f"[Client] Retry to start the login process. Retry count: {retry_count}")
