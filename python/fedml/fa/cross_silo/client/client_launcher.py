import subprocess
from fedml.arguments import load_arguments
from fedml.constants import FEDML_TRAINING_PLATFORM_CROSS_SILO


class CrossSiloLauncher:
    @staticmethod
    def launch_dist_trainers(torch_client_filename, inputs):
        # this is only used by the client (DDP or single process), so there is no need to specify the backend.
        args = load_arguments(FEDML_TRAINING_PLATFORM_CROSS_SILO)
        CrossSiloLauncher._run_cross_silo_horizontal(args, torch_client_filename, inputs)

    @staticmethod
    def _run_cross_silo_horizontal(args, torch_client_filename, inputs):
        python_path = subprocess.run(["which", "python"], capture_output=True, text=True).stdout.strip()
        process_arguments = [python_path, torch_client_filename] + inputs
        subprocess.run(process_arguments)