import os

import boto3

from python.fedml.utils import logging
from python.fedml.utils.logging import logger

BUCKET_NAME = "fedmls3"
# aws_access_key_id、aws_secret_access_key、region
CN_S3_AKI = ''
CN_S3_SAK = ''
CN_REGION_NAME = 'us-east-1'
# s3 client
s3 = boto3.client('s3', region_name=CN_REGION_NAME,
                  aws_access_key_id=CN_S3_AKI, aws_secret_access_key=CN_S3_SAK)


@logger.catch
def upload_file(src_local_path, dest_s3_path):
    """
    upload file
    :param src_local_path:
    :param dest_s3_path:
    :return:
    """
    try:
        with open(src_local_path, 'rb') as f:
            s3.upload_fileobj(f, BUCKET_NAME, dest_s3_path)
    except Exception as e:
        logging.error(f'Upload data failed. | src: {src_local_path} | dest: {dest_s3_path} | Exception: {e}')
        return False
    logging.info(f'Uploading file successful. | src: {src_local_path} | dest: {dest_s3_path}')
    return True


def download_file(path_s3, path_local):
    """
    download file
    :param path_s3: s3 key
    :param path_local: local path
    :return:
    """
    retry = 0
    while retry < 3:
        logging.info(f'Start downloading files. | path_s3: {path_s3} | path_local: {path_local}')
        try:
            s3.download_file(BUCKET_NAME, path_s3, path_local)
            file_size = os.path.getsize(path_local)
            logging.info(f'Downloading completed. | size: {round(file_size / 1048576, 2)} MB')
            break
        except Exception as e:
            logging.error(f'Download zip failed. | Exception: {e}')
            retry += 1
    if retry >= 3:
        logging.error(f'Download zip failed after max retry.')


@logger.catch
def delete_s3_zip(path_s3):
    """
    delete s3 object
    :param path_s3: s3 key
    :return:
    """
    s3.delete_object(Bucket=BUCKET_NAME, Key=path_s3)
    logging.info(f'Delete s3 file Successful. | path_s3 = {path_s3}')


if __name__ == "__main__":
    upload_file("./s3_test_file", "run_id_000001/client_id_s3_test_file")
