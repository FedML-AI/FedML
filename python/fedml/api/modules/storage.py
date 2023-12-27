import logging
import os
import shutil
from fedml.api.modules.utils import authenticate
from fedml.core.distributed.communication.s3.remote_storage import S3Storage
from fedml.core.mlops.mlops_configs import Configs, MLOpsConfigs
from fedml.computing.scheduler.master.server_constants import ServerConstants


def upload(data_path, api_key, name, show_progress, out_progress_to_err, progress_desc, metadata) -> str:
    api_key = authenticate(api_key)
    user_id = _get_user_id_from_api_key(api_key)
    if user_id is None:
        print(f"Failed to get user id from api key: {api_key}")
        return None
    configs = MLOpsConfigs.fetch_remote_storage_configs()
    r2_config = configs[Configs.R2_CONFIG]
    s3 = S3Storage(r2_config)
    src_local_path = os.path.abspath(data_path)
    root_dir = os.path.dirname(src_local_path)
    archive_name = os.path.basename(src_local_path) + '.zip'
    archive_path = os.path.join(root_dir, archive_name)
    try:
        # Create a zip archive
        shutil.make_archive(archive_path[:-4], 'zip', root_dir=root_dir, base_dir=os.path.basename(src_local_path))

        # Now you can use the archive_path for further processing or uploading
        print(f"Successfully archived data at: {archive_path}")
    except Exception as e:
        print(f"Error archiving data: {e}")
        return None

    file_name = (os.path.basename(archive_path) if name is None else name) + ".zip"
    dest_path = os.path.join(user_id, file_name)
    file_uploaded_url = s3.upload_file_with_progress(src_local_path=archive_path, dest_s3_path=dest_path,
                                                     show_progress=show_progress,
                                                     out_progress_to_err=out_progress_to_err,
                                                     progress_desc=progress_desc, metadata=metadata)
    os.remove(archive_path)
    return file_uploaded_url


def download(data_name, api_key=None, dest_path=None):
    api_key = authenticate(api_key)
    user_id = _get_user_id_from_api_key(api_key)
    if user_id is None:
        print(f"Failed to get user id from api key: {api_key}")
        return None
    configs = MLOpsConfigs.fetch_remote_storage_configs()
    r2_config = configs[Configs.R2_CONFIG]
    s3 = S3Storage(r2_config)
    zip_file_name = data_name + ".zip"
    key = os.path.join(user_id, zip_file_name)
    path_local = os.path.abspath(zip_file_name)
    dest_path = os.path.abspath(dest_path) if dest_path else data_name
    if s3.download_file_with_progress(path_s3=key, path_local=path_local):
        try:
            shutil.unpack_archive(path_local, dest_path)
            os.remove(path_local)
            return os.path.abspath(dest_path)
        except Exception as e:
            logging.error(f"Failed to unpack archive: {e}")
            return None
    else:
        logging.error(f"Failed to download data: {data_name}")
        return None


def get_metadata(data_name, api_key=None):
    api_key = authenticate(api_key)
    user_id = _get_user_id_from_api_key(api_key)
    if user_id is None:
        print(f"Failed to get user id from api key: {api_key}")
        return None
    configs = MLOpsConfigs.fetch_remote_storage_configs()
    r2_config = configs[Configs.R2_CONFIG]
    s3 = S3Storage(r2_config)
    zip_file_name = data_name + ".zip"
    path_s3 = os.path.join(user_id, zip_file_name)
    return s3.get_object_metadata(path_s3=path_s3)


def _get_user_id_from_api_key(api_key: str) -> str:
    user_url = ServerConstants.get_user_url()
    json_data = {
        "apiKey": api_key
    }
    response = ServerConstants.request(user_url, json_data)
    if response.status_code != 200:
        print(f"Failed to get user info with response.status_code = {response.status_code}, "
              f"response.content: {response.content}")
        return None
    else:
        resp_data = response.json()
        code = resp_data.get("code", None)
        data = resp_data.get("data", None)
        if code is None or data is None or code == "FAILURE":
            message = resp_data.get("message", None)
            print(f"Failed querying user info with following meesage: {message}"
                  f"response.content: {response.content}")
            return None
    user_id = data.get("id", None)
    return str(user_id) if user_id else None
