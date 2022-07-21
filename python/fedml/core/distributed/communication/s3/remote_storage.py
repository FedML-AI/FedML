import logging
import os
import pickle
import time

import boto3
import yaml

from fedml.core.mlops.mlops_profiler_event import MLOpsProfilerEvent

# for multi-processing, we need to create a global variable for AWS S3 client:
# https://www.pythonforthelab.com/blog/differences-between-multiprocessing-windows-and-linux/
# https://stackoverflow.com/questions/72313845/multiprocessing-picklingerror-cant-pickle-class-botocore-client-s3-attr
aws_s3_client = None
aws_s3_resource = None


class S3Storage:
    def __init__(self, s3_config_path):
        self.bucket_name = None
        self.cn_region_name = None
        self.cn_s3_sak = None
        self.cn_s3_aki = None
        self.set_config_from_file(s3_config_path)
        self.set_config_from_objects(s3_config_path)
        global aws_s3_client
        aws_s3_client = boto3.client(
            "s3",
            region_name=self.cn_region_name,
            aws_access_key_id=self.cn_s3_aki,
            aws_secret_access_key=self.cn_s3_sak,
        )

        global aws_s3_resource
        aws_s3_resource = boto3.resource(
            "s3",
            region_name=self.cn_region_name,
            aws_access_key_id=self.cn_s3_aki,
            aws_secret_access_key=self.cn_s3_sak,
        )

    def write_model(self, message_key, model):
        global aws_s3_client
        pickle_dump_start_time = time.time()
        model_pkl = pickle.dumps(model)
        MLOpsProfilerEvent.log_to_wandb(
            {"PickleDumpsTime": time.time() - pickle_dump_start_time}
        )
        s3_upload_start_time = time.time()
        aws_s3_client.put_object(
            Body=model_pkl, Bucket=self.bucket_name, Key=message_key, ACL="public-read",
        )
        MLOpsProfilerEvent.log_to_wandb(
            {"Comm/send_delay": time.time() - s3_upload_start_time}
        )
        model_url = aws_s3_client.generate_presigned_url(
            "get_object",
            ExpiresIn=60 * 60 * 24 * 5,
            Params={"Bucket": self.bucket_name, "Key": message_key},
        )
        return model_url

    def read_model(self, message_key):
        global aws_s3_client
        message_handler_start_time = time.time()
        obj = aws_s3_client.get_object(Bucket=self.bucket_name, Key=message_key)
        model_pkl = obj["Body"].read()
        MLOpsProfilerEvent.log_to_wandb(
            {"Comm/recieve_delay_s3": time.time() - message_handler_start_time}
        )
        unpickle_start_time = time.time()
        model = pickle.loads(model_pkl)
        MLOpsProfilerEvent.log_to_wandb(
            {"UnpickleTime": time.time() - unpickle_start_time}
        )
        return model

    # def write_model(self, message_key, model):
    #     with tempfile.TemporaryFile() as fp:
    #         joblib.dump(model, fp)
    #         fp.seek(0)
    #         global aws_s3_client
    #         aws_s3_client.put_object(
    #             Body=fp.read(),
    #             Bucket=self.bucket_name,
    #             Key=message_key,
    #             ACL="public-read",
    #         )
    #         model_url = aws_s3_client.generate_presigned_url(
    #             "get_object",
    #             ExpiresIn=60 * 60 * 24 * 5,
    #             Params={"Bucket": self.bucket_name, "Key": message_key},
    #         )
    #         return model_url

    # def read_model(self, message_key):
    #     with tempfile.TemporaryFile() as fp:
    #         global aws_s3_client
    #         aws_s3_client.download_fileobj(
    #             Fileobj=fp, Bucket=self.bucket_name, Key=message_key
    #         )
    #         fp.seek(0)
    #         try:
    #             model = joblib.load(fp)
    #         except Exception as e:
    #             print("Exception " + str(e))
    #     return model

    def upload_file(self, src_local_path, dest_s3_path):
        """
        upload file
        :param src_local_path:
        :param dest_s3_path:
        :return:
        """
        try:
            with open(src_local_path, "rb") as f:
                global aws_s3_client
                aws_s3_client.upload_fileobj(
                    f, self.bucket_name, dest_s3_path, ExtraArgs={"ACL": "public-read"}
                )
        except Exception as e:
            logging.error(
                f"Upload data failed. | src: {src_local_path} | dest: {dest_s3_path} | Exception: {e}"
            )
            return False
        logging.info(
            f"Uploading file successful. | src: {src_local_path} | dest: {dest_s3_path}"
        )
        return True

    def download_file(self, path_s3, path_local):
        """
        download file
        :param path_s3: s3 key
        :param path_local: local path
        :return:
        """
        retry = 0
        while retry < 3:
            logging.info(
                f"Start downloading files. | path_s3: {path_s3} | path_local: {path_local}"
            )
            try:
                global aws_s3_client
                aws_s3_client.download_file(self.bucket_name, path_s3, path_local)
                file_size = os.path.getsize(path_local)
                logging.info(
                    f"Downloading completed. | size: {round(file_size / 1048576, 2)} MB"
                )
                break
            except Exception as e:
                logging.error(f"Download zip failed. | Exception: {e}")
                retry += 1
        if retry >= 3:
            logging.error(f"Download zip failed after max retry.")

    def delete_s3_zip(self, path_s3):
        """
        delete s3 object
        :param path_s3: s3 key
        :return:
        """
        global aws_s3_client
        aws_s3_client.delete_object(Bucket=self.bucket_name, Key=path_s3)
        logging.info(f"Delete s3 file Successful. | path_s3 = {path_s3}")

    def set_config_from_file(self, config_file_path):
        try:
            with open(config_file_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                self.cn_s3_aki = config["CN_S3_AKI"]
                self.cn_s3_sak = config["CN_S3_SAK"]
                self.cn_region_name = config["CN_REGION_NAME"]
                self.bucket_name = config["BUCKET_NAME"]
        except Exception as e:
            pass

    def set_config_from_objects(self, s3_config):
        self.cn_s3_aki = s3_config["CN_S3_AKI"]
        self.cn_s3_sak = s3_config["CN_S3_SAK"]
        self.cn_region_name = s3_config["CN_REGION_NAME"]
        self.bucket_name = s3_config["BUCKET_NAME"]
