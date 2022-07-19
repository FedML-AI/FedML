from fedml.cross_silo.hierarchical.client_launcher import launch_dist_trainers

torch_client_filename = "torch_client.py"

launch_dist_trainers(torch_client_filename)
