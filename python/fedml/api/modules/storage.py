import logging
import os
import shutil
from fedml.api.modules.utils import authenticate
from fedml.core.distributed.communication.s3.remote_storage import S3Storage
from fedml.core.mlops.mlops_configs import Configs, MLOpsConfigs


def upload(data_path, api_key, name, show_progress, out_progress_to_err, progress_desc) -> str:
    authenticate(api_key)
    configs = MLOpsConfigs.fetch_remote_storage_configs()
    r2_config = configs[Configs.R2_CONFIG]
    # Todo: Delete following
    # r2_config = {'BUCKET_NAME': 'dataset',
    #              'CN_S3_ENDPOINT': 'https://03aa47c68e20656e11ca9e0765c6bc1f.r2.cloudflarestorage.com',
    #              'CN_S3_AKI': '44b582b7fe45ec884ec011d34f1da78d',
    #              'CN_S3_SAK': '41ea9ff0536ddf35daf5963da552a6c576c3a66e2359dffe148cf3162d8b2de6'}
    s3 = S3Storage(r2_config)
    src_local_path = os.path.abspath(data_path)
    root_dir = os.path.dirname(src_local_path)
    file_name = os.path.basename(src_local_path) if name is None else name
    archive_name = file_name + '.zip'
    archive_path = os.path.join(root_dir, archive_name)
    try:
        # Create a zip archive
        shutil.make_archive(archive_path[:-4], 'zip', root_dir=root_dir, base_dir=os.path.basename(src_local_path))

        # Now you can use the archive_path for further processing or uploading
        print(f"Successfully archived data at: {archive_path}")
    except Exception as e:
        print(f"Error archiving data: {e}")
        return None
    dest_path = os.path.join(api_key, os.path.basename(archive_path))
    file_uploaded_url = s3.upload_file_with_progress(src_local_path=archive_path, dest_s3_path=dest_path,
                                                     show_progress=show_progress,
                                                     out_progress_to_err=out_progress_to_err,
                                                     progress_desc=progress_desc)
    os.remove(archive_path)
    return file_uploaded_url


def download(data_name, api_key) -> str:
    authenticate(api_key)
    configs = MLOpsConfigs.fetch_remote_storage_configs()
    r2_config = configs[Configs.R2_CONFIG]
    # Todo: Delete following
    # r2_config = {'BUCKET_NAME': 'dataset',
    #              'CN_S3_ENDPOINT': 'https://03aa47c68e20656e11ca9e0765c6bc1f.r2.cloudflarestorage.com',
    #              'CN_S3_AKI': '44b582b7fe45ec884ec011d34f1da78d',
    #              'CN_S3_SAK': '41ea9ff0536ddf35daf5963da552a6c576c3a66e2359dffe148cf3162d8b2de6'}
    s3 = S3Storage(r2_config)
    zip_file_name = data_name + ".zip"
    key = os.path.join(api_key, zip_file_name)
    path_local = os.path.abspath(zip_file_name)
    if s3.download_file_with_progress(path_s3=key, path_local=path_local):
        try:
            shutil.unpack_archive(path_local, data_name)
            os.remove(path_local)
            return os.path.abspath(data_name)
        except Exception as e:
            logging.error(f"Failed to unpack archive: {e}")
            return None
    else:
        logging.error(f"Failed to download data: {data_name}")
        return None
