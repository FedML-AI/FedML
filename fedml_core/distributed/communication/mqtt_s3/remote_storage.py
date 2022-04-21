import os
import tempfile
import uuid

import joblib
import yaml
import boto3
from botocore.retries import bucket
from loguru import logger
import logging
import json


class S3Storage():
    def __init__(self, s3_config_path):

        self.set_config_from_file(s3_config_path)
        self.s3 = boto3.client('s3', region_name=self.cn_region_name,
                               aws_access_key_id=self.cn_s3_aki, aws_secret_access_key=self.cn_s3_sak)

        self.s3_resource = boto3.resource('s3', region_name=self.cn_region_name,
                                          aws_access_key_id=self.cn_s3_aki, aws_secret_access_key=self.cn_s3_sak)

    def write_json(self, message_key, payload):
        obj = self.s3_resource.Object(self.bucket_name, message_key)
        obj.put(Body=payload)

    def read_json(self, message_key):
        obj = self.s3_resource.Object(self.bucket_name, message_key)
        payload = obj.get()['Body'].read()
        return payload

    def write_model(self, message_key, model):
        with tempfile.TemporaryFile() as fp:
            joblib.dump(model, fp)
            fp.seek(0)
            self.s3.put_object(Body=fp.read(), Bucket=self.bucket_name, Key=message_key, ACL='public-read')
            model_url = self.s3.generate_presigned_url(
                "get_object", ExpiresIn=60*60*24*5, Params={"Bucket": self.bucket_name, "Key": message_key}
            )
            return model_url

    def write_model_with_file(self, message_key, model_file):
        self.upload_file(model_file, message_key)
        model_url = self.s3.generate_presigned_url(
            "get_object", ExpiresIn=60*60*24*5, Params={"Bucket": self.bucket_name, "Key": message_key}
        )
        return model_url

    def read_model(self, message_key):
        with tempfile.TemporaryFile() as fp:
            self.s3.download_fileobj(Fileobj=fp, Bucket=self.bucket_name, Key=message_key)
            fp.seek(0)
            try:
                model = joblib.load(fp)
            except Exception as e:
                print("Exception "+str(e))
        return model

    def read_model_with_file(self, message_key):
        model_file = "./" + str(uuid.uuid4()) + ".mnn"
        self.download_file(message_key, model_file)
        return model_file

    @logger.catch
    def upload_file(self, src_local_path, dest_s3_path):
        """
        upload file
        :param src_local_path:
        :param dest_s3_path:
        :return:
        """
        try:

            with open(src_local_path, 'rb') as f:
                self.s3.upload_fileobj(f, self.bucket_name, dest_s3_path, ExtraArgs={'ACL': 'public-read'})
        except Exception as e:
            logging.error(
                f'Upload data failed. | src: {src_local_path} | dest: {dest_s3_path} | Exception: {e}')
            return False
        logging.info(
            f'Uploading file successful. | src: {src_local_path} | dest: {dest_s3_path}')
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
            # 下载异常尝试3次
            logging.info(
                f'Start downloading files. | path_s3: {path_s3} | path_local: {path_local}')
            try:
                self.s3.download_file(self.bucket_name, path_s3, path_local)
                file_size = os.path.getsize(path_local)
                logging.info(
                    f'Downloading completed. | size: {round(file_size / 1048576, 2)} MB')
                # 下载完成后退出重试
                break
            except Exception as e:
                logging.error(f'Download zip failed. | Exception: {e}')
                retry += 1
        if retry >= 3:
            logging.error(f'Download zip failed after max retry.')

    @logger.catch
    def delete_s3_zip(self, path_s3):
        """
        delete s3 object
        :param path_s3: s3 key
        :return:
        """
        self.s3.delete_object(Bucket=self.bucket_name, Key=path_s3)
        logging.info(f'Delete s3 file Successful. | path_s3 = {path_s3}')

    def set_config_from_file(self, config_file_path):
        with open(config_file_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.cn_s3_aki = config['CN_S3_AKI']
            self.cn_s3_sak = config['CN_S3_SAK']
            self.cn_region_name = config['CN_REGION_NAME']
            self.bucket_name = config['BUCKET_NAME']

# if __name__ == "__main__":
#     upload_file("./s3_test_file", "run_id_000001/client_id_s3_test_file")
