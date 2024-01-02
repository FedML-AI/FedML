import sys
from fedml.cross_silo.client.client_launcher import CrossSiloLauncher

if __name__ == "__main__":
    CrossSiloLauncher.launch_dist_trainers("main_fedml_cross_silo_hi.py", sys.argv[1:])