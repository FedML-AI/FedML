import os
import sys
from pathlib import Path
import subprocess

if __name__ == '__main__':
    curr_dir = Path(__file__).parent
    os.chdir(curr_dir)

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
