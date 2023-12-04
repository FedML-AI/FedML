from fedml.api.modules.utils import authenticate

def upload(dataset_path, name, api_key):
    authenticate(api_key)
    FedMLDatasetManager.get_instance().upload_dataset(api_key, fedml.get_env_version(), dataset_path, name,
