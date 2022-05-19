import yaml


def load_yaml_config(yaml_path):
    """Helper function to load a yaml config file"""
    with open(yaml_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError("Yaml error - check yaml file")
