from .remote_storage import S3Storage


class S3MNNStorage:
    def __init__(self, s3_config_path):
        self.s3_storage = S3Storage(s3_config_path)

    # def write_json(self, message_key, payload):
    #     self.s3_storage.write_json(message_key, payload)

    # def read_json(self, message_key):
    #     return self.s3_storage.read_json(message_key)

    def upload_model_file(self, message_key, model_file_path):
        """
        Uploads a model file to S3 storage for Mobile Platform (MNN).

        Args:
            message_key (str): The key to identify the uploaded model in S3.
            model_file_path (str): The local file path of the model to be uploaded.

        Returns:
            bool: True if the upload was successful, False otherwise.
        """
        return self.s3_storage.upload_file(model_file_path, message_key)

    def download_model_file(self, message_key, model_file_path):
        """
        Downloads a model file from S3 storage for Mobile Platform (MNN).

        Args:
            message_key (str): The key identifying the model to be downloaded from S3.
            model_file_path (str): The local file path where the downloaded model will be saved.

        Returns:
            None
        """
        self.s3_storage.download_file(message_key, model_file_path)

    def write_model(self, message_key, model):
        """
        Writes a model object to S3 storage.

        Args:
            message_key (str): The key to identify the stored model in S3.
            model: The model object to be stored.

        Returns:
            None
        """
        self.s3_storage.write_model(message_key, model)

    def read_model(self, message_key):
        """
        Reads a model object from S3 storage.

        Args:
            message_key (str): The key identifying the model to be read from S3.

        Returns:
            object: The model object read from S3.
        """
        return self.s3_storage.read_model(message_key)

    def upload_file(self, src_local_path, dest_s3_path):
        """
        Uploads a file from the local system to S3 storage.

        Args:
            src_local_path (str): The local file path of the file to be uploaded.
            dest_s3_path (str): The S3 destination path for the uploaded file.

        Returns:
            bool: True if the upload was successful, False otherwise.
        """
        return self.s3_storage.upload_file(src_local_path, dest_s3_path)

    def download_file(self, path_s3, path_local):
        """
        Downloads a file from S3 storage to the local system.

        Args:
            path_s3 (str): The S3 path of the file to be downloaded.
            path_local (str): The local file path where the downloaded file will be saved.

        Returns:
            None
        """
        self.s3_storage.download_file(path_s3, path_local)

    def delete_s3_zip(self, path_s3):
        """
        Deletes a ZIP file from S3 storage.

        Args:
            path_s3 (str): The S3 path of the ZIP file to be deleted.

        Returns:
            None
        """
        self.s3_storage.delete_s3_zip(path_s3)

    def set_config_from_file(self, config_file_path):
        """
        Sets the S3 configuration from a configuration file.

        Args:
            config_file_path (str): The path to the S3 configuration file.

        Returns:
            None
        """
        self.s3_storage.set_config_from_file(config_file_path)

    def set_config_from_objects(self, s3_config):
        """
        Sets the S3 configuration from configuration objects.

        Args:
            s3_config: Configuration objects for S3 storage.

        Returns:
            None
        """
        self.s3_storage.set_config_from_objects(s3_config)
