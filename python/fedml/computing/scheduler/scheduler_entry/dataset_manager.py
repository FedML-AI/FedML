from fedml.core.common.singleton import Singleton

import fedml
from fedml.core.distributed.communication.s3.remote_storage import S3Storage


class FedMLDatasetManager(Singleton):

    def __init__(self):
        self.config_version = fedml.get_env_version()
        _, r2_config = fedml.get
        self.remote_storage = S3Storage()

    @staticmethod
    def get_instance():
        return FedMLDatasetManager()

    def upload_dataset(self, api_key, version, path, dataset_name, show_progress=False, out_progress_to_err=False,
                       progress_desc=None):
        fedml.set_env_version(version)

        pass
