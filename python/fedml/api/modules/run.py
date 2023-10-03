import fedml.api
from fedml.api.modules.utils import authenticate


def command(cmd, cluster_name, api_key):
    authenticate(api_key)
    if not fedml.api.cluster_exists(cluster_name):
        raise Exception(f"Cluster {cluster_name} does not exist. Run can only be executed on clusters that already "
                        f"exists.")
    return fedml.api.launch_job(yaml_file=cmd, cluster=cluster_name, api_key=api_key)
