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
        this is used for Mobile Platform (MNN)
        :param message_key:
        :param model_file_path:
        :return:
        """
        return self.s3_storage.upload_file(model_file_path, message_key)

    def download_model_file(self, message_key, model_file_path):
        """
        this is used for Mobile Platform (MNN)
        :param message_key:
        :param model_file_path:
        :return:
        """
        self.s3_storage.download_file(message_key, model_file_path)

    def write_model(self, message_key, model):
        self.s3_storage.write_model(message_key, model)

    def read_model(self, message_key):
        return self.s3_storage.read_model(message_key)

    def upload_file(self, src_local_path, dest_s3_path):
        return self.s3_storage.upload_file(src_local_path, dest_s3_path)

    def download_file(self, path_s3, path_local):
        self.s3_storage.download_file(path_s3, path_local)

    def delete_s3_zip(self, path_s3):
        self.s3_storage.delete_s3_zip(path_s3)

    def set_config_from_file(self, config_file_path):
        self.s3_storage.set_config_from_file(config_file_path)

    def set_config_from_objects(self, s3_config):
        self.s3_storage.set_config_from_objects(s3_config)
