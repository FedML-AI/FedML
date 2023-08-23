import logging
import os
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)

    # go to project root directory
    os.chdir(Path(__file__).parent)

    torch_distributed_default_port = int(os.getenv("TORCH_DISTRIBUTED_DEFAULT_PORT", 29500))
    args, *_ = parser.parse_known_args()
    master_port = torch_distributed_default_port + args.rank

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
    print(f"sys.argv = {sys.argv}")
    result = subprocess.run(
        " ".join([
            "bash",
            "scripts/run_fedml.sh",
            "\"\"",  # master address
            f"{master_port}",  # master port
            "\"\"",  # number of nodes
            "main_fedllm.py",  # main program
            *sys.argv[1:],
        ]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )

    logging.info(result.stdout)
    logging.error(result.stderr)

    exit(result.returncode)
