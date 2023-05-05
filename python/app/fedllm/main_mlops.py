import os
import sys
from pathlib import Path
import subprocess

if __name__ == '__main__':
    curr_dir = Path(__file__).parent
    os.chdir(curr_dir)

    """
        process = ClientConstants.exec_console_with_shell_script_list(
            [
                python_program,         # python
                entry_fill_full_path,   # ./main_mlops.py
                "--cf",                 # --cf
                conf_file_full_path,    # $mlops_path/fedml_config.yaml
                "--rank",               # --rank
                str(dynamic_args_config["rank"]), # rank
                "--role",               # --role
                "client",               # client
            ],
        python main_mlops.py --cf fedml_config/fedml_config.yaml --rank 0 --role client
    """
    print("sys.argv = {}".format(sys.argv))
    result = subprocess.run(
        " ".join([
            "bash",
            "scripts/run_fedml.sh",
            "\"\"",  # master address
            "\"\"",  # master port
            "\"\"",  # number of nodes
            "main_fedllm.py",  # main program
            *sys.argv[1:],
        ]),
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        stdout=sys.stdout,
        stderr=sys.stderr,
        shell=True,
        # capture_output=True,
        # text=True
    )
    print(result)
