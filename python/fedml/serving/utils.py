import os
import logging
import traceback
from ..computing.scheduler.model_scheduler.device_client_constants import ClientConstants
from ..computing.scheduler.comm_utils import sys_utils
import platform
import stat

def run_bootstrap(bootstrap_script_path, bootstrap_script_file):
    is_bootstrap_run_ok = True
    try:
        if bootstrap_script_path is not None:
            if os.path.exists(bootstrap_script_path):
                bootstrap_stat = os.stat(bootstrap_script_path)
                if platform.system() == 'Windows':
                    os.chmod(bootstrap_script_path,
                                bootstrap_stat.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    bootstrap_scripts = "{}".format(bootstrap_script_path)
                else:
                    os.chmod(bootstrap_script_path,
                                bootstrap_stat.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                    bootstrap_scripts = "cd {}; sh {}".format(bootstrap_script_path, # Use sh over ./ to avoid permission denied error
                                                                os.path.basename(bootstrap_script_file))
                bootstrap_scripts = str(bootstrap_scripts).replace('\\', os.sep).replace('/', os.sep)

                process = ClientConstants.exec_console_with_script(bootstrap_scripts, should_capture_stdout=True,
                                                                    should_capture_stderr=True)
                # ClientConstants.save_bootstrap_process(run_id, process.pid)
                ret_code, out, err = ClientConstants.get_console_pipe_out_err_results(process)

                if ret_code is None or ret_code <= 0:
                    if out is not None:
                        out_str = sys_utils.decode_our_err_result(out)
                        if out_str != "":
                            logging.info("{}".format(out_str))

                    sys_utils.log_return_info(bootstrap_script_file, 0)

                    is_bootstrap_run_ok = True
                else:
                    if err is not None:
                        err_str = sys_utils.decode_our_err_result(err)
                        if err_str != "":
                            logging.error("{}".format(err_str))

                    sys_utils.log_return_info(bootstrap_script_file, ret_code)

                    is_bootstrap_run_ok = False
    except Exception as e:
        logging.error("Bootstrap script error: {}".format(traceback.format_exc()))
        is_bootstrap_run_ok = False

    return is_bootstrap_run_ok
