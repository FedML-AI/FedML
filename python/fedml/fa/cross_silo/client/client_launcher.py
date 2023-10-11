import subprocess
from fedml.arguments import load_arguments
from fedml.constants import FEDML_TRAINING_PLATFORM_CROSS_SILO

class CrossSiloLauncher:
    """
    A class for launching distributed trainers in a cross-silo federated learning setup.

    Attributes:
        None

    Methods:
        launch_dist_trainers(torch_client_filename, inputs):
            Launch distributed trainers using the provided arguments.

    """
    @staticmethod
    def launch_dist_trainers(torch_client_filename, inputs):
        """
        Launch distributed trainers using the provided arguments.

        Args:
            torch_client_filename (str): The filename of the PyTorch client script.
            inputs (list): A list of input arguments to be passed to the client script.

        Returns:
            None
        """
        # This is only used by the client (DDP or single process), so there is no need to specify the backend.
        args = load_arguments(FEDML_TRAINING_PLATFORM_CROSS_SILO)
        CrossSiloLauncher._run_cross_silo_horizontal(args, torch_client_filename, inputs)

    @staticmethod
    def _run_cross_silo_horizontal(args, torch_client_filename, inputs):
        """
        Run the cross-silo horizontal federated learning process.

        Args:
            args: Configuration arguments.
            torch_client_filename (str): The filename of the PyTorch client script.
            inputs (list): A list of input arguments to be passed to the client script.

        Returns:
            None
        """
        python_path = subprocess.run(["which", "python"], capture_output=True, text=True).stdout.strip()
        process_arguments = [python_path, torch_client_filename] + inputs
        subprocess.run(process_arguments)
