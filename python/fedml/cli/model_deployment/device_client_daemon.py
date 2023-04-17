
import argparse
import os
import time
import sys

from fedml.cli.comm_utils import sys_utils
from fedml.cli.model_deployment.device_client_constants import ClientConstants


def ota_upgrade(in_args):
    should_upgrade = False
    fedml_is_latest_version = True
    try:
        fedml_is_latest_version, local_ver, remote_ver = sys_utils.check_fedml_is_latest_version(in_args.version)
        should_upgrade = False if fedml_is_latest_version else True
    except Exception as e:
        return

    if not should_upgrade:
        return
    upgrade_version = remote_ver

    python_ver_major = sys.version_info[0]
    python_ver_minor = sys.version_info[1]
    if in_args.version == "release":
        if python_ver_major == 3 and python_ver_minor == 7:
            os.system(f"pip uninstall -y fedml;pip install fedml=={upgrade_version} --use-deprecated=legacy-resolver")
        else:
            os.system(f"pip uninstall -y fedml;pip install fedml=={upgrade_version}")
    else:
        if python_ver_major == 3 and python_ver_minor == 7:
            os.system(f"pip uninstall -y fedml;"
                      f"pip install --index-url https://test.pypi.org/simple/ "
                      f"--extra-index-url https://pypi.org/simple fedml=={upgrade_version} "
                      f"--use-deprecated=legacy-resolver")
        else:
            os.system(f"pip uninstall -y fedml;"
                      f"pip install --index-url https://test.pypi.org/simple/ "
                      f"--extra-index-url https://pypi.org/simple fedml=={upgrade_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--type", "-t", help="Login or logout to ModelOps platform")
    parser.add_argument("--user", "-u", type=str,
                        help='account id at ModelOps platform')
    parser.add_argument("--version", "-v", type=str, default="release")
    parser.add_argument("--local_server", "-ls", type=str, default="127.0.0.1")
    parser.add_argument("--role", "-r", type=str, default="client")
    parser.add_argument("--device_id", "-id", type=str, default="0")
    parser.add_argument("--os_name", "-os", type=str, default="")
    parser.add_argument("--infer_host", "-ih", type=str, default="127.0.0.1")
    args = parser.parse_args()
    args.user = args.user

    pip_source_dir = os.path.dirname(__file__)
    login_cmd = os.path.join(pip_source_dir, "device_client_login.py")
    while True:
        try:
            ClientConstants.cleanup_run_process()
            sys_utils.cleanup_all_fedml_client_api_processes(is_model_device=True)
            sys_utils.cleanup_all_fedml_client_learning_processes()
            sys_utils.cleanup_all_fedml_client_login_processes("device_client_login.py", clean_process_group=False)
        except Exception as e:
            pass

        ota_upgrade(args)

        login_pid = ClientConstants.exec_console_with_shell_script_list(
            [
                sys_utils.get_python_program(),
                "-W",
                "ignore",
                login_cmd,
                "-t",
                "login",
                "-u",
                args.user,
                "-v",
                args.version,
                "-ls",
                args.local_server,
                "-r",
                args.role,
                "-id",
                args.device_id,
                "-os",
                args.os_name,
                "-ih",
                args.infer_host
            ]
        )
        ret_code, exec_out, exec_err = ClientConstants.get_console_sys_out_pipe_err_results(login_pid)
        time.sleep(3)

