import logging
import os
import shutil

import requests
from fedml.api.modules.utils import authenticate
from fedml.core.distributed.communication.s3.remote_storage import S3Storage
from fedml.core.mlops.mlops_configs import Configs, MLOpsConfigs
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.api.fedml_response import FedMLResponse, ResponseCode


# Todo (alaydshah): Add file size
class StorageMetadata(object):
    def __init__(self, data: dict):
        self.dataName = data.get("datasetName", None)
        self.description = data.get("description", None)
        self.tags = data.get("description", None)
        self.createdAt = data.get("createTime", None)
        self.updatedAt = data.get("updateTime", None)


# Todo (alaydshah): Add file size while creating objects. Store service name in metadata
# Todo (alaydshah): If data already exists, don't upload again. Instead suggest to use update command


def upload(data_path, api_key, name, description, service, show_progress, out_progress_to_err, progress_desc,
           metadata) -> FedMLResponse:
    api_key = authenticate(api_key)

    user_id, message = _get_user_id_from_api_key(api_key)

    if user_id is None:
        return FedMLResponse(code=ResponseCode.FAILURE, message=message)

    archive_path, message = _archive_data(data_path)
    if not archive_path:
        return FedMLResponse(code=ResponseCode.FAILURE, message=message)

    store = _get_storage_service(service)
    name = os.path.splitext(os.path.basename(archive_path))[0] if name is None else name
    file_name = name + ".zip"
    dest_path = os.path.join(user_id, file_name)

    file_uploaded_url = store.upload_file_with_progress(src_local_path=archive_path, dest_s3_path=dest_path,
                                                        show_progress=show_progress,
                                                        out_progress_to_err=out_progress_to_err,
                                                        progress_desc=progress_desc, metadata=metadata)
    os.remove(archive_path)
    if not file_uploaded_url:
        return FedMLResponse(code=ResponseCode.FAILURE, message=f"Failed to upload file: {archive_path}")

    json_data = {
        "datasetName": name,
        "description": description,
        "fileSize": "",
        "fileUrl": file_uploaded_url,
        "tagNameList": [],
    }

    try:
        response = _create_dataset(api_key=api_key, json_data=json_data)
        code, message, data = _get_data_from_response(message="Failed to upload data", response=response)
    except Exception as e:
        return FedMLResponse(code=ResponseCode.FAILURE, message=f"Failed to create dataset: {e}")

    if data:
        return FedMLResponse(code=code, message=message, data=file_uploaded_url)
    return FedMLResponse(code=code, message=message)


# Todo(alaydshah): Query service from object metadata
def download(data_name, api_key, service, dest_path, show_progress=True) -> FedMLResponse:
    api_key = authenticate(api_key)
    user_id, message = _get_user_id_from_api_key(api_key)

    if user_id is None:
        return FedMLResponse(code=ResponseCode.FAILURE, message=message)

    store = _get_storage_service(service)
    zip_file_name = data_name + ".zip"
    key = os.path.join(user_id, zip_file_name)
    path_local = os.path.abspath(zip_file_name)
    dest_path = os.path.abspath(dest_path) if dest_path else data_name
    if store.download_file_with_progress(path_s3=key, path_local=path_local, show_progress=show_progress):
        try:
            shutil.unpack_archive(path_local, dest_path)
            os.remove(path_local)
            abs_dest_path = os.path.abspath(dest_path)
            return FedMLResponse(code=ResponseCode.SUCCESS, message=f"Successfully downloaded and unzipped data at "
                                                                    f"{abs_dest_path}", data=abs_dest_path)
        except Exception as e:
            error_message = f"Failed to unpack archive: {e}"
            logging.error(error_message)
            return FedMLResponse(code=ResponseCode.FAILURE, message=error_message)
    else:
        error_message = f"Failed to download data: {data_name}"
        logging.error(error_message)
        return FedMLResponse(code=ResponseCode.FAILURE, message=error_message)


def get_user_metadata(data_name, api_key=None) -> FedMLResponse:
    api_key = authenticate(api_key)
    user_id, message = _get_user_id_from_api_key(api_key)

    if user_id is None:
        return FedMLResponse(code=ResponseCode.FAILURE, message=message)

    configs = MLOpsConfigs.fetch_remote_storage_configs()
    r2_config = configs[Configs.R2_CONFIG]
    store = S3Storage(r2_config)
    zip_file_name = data_name + ".zip"
    path_s3 = os.path.join(user_id, zip_file_name)
    data, message = store.get_object_metadata(path_s3=path_s3)

    # Data can be {} if no metadata exists. That should be a valid response. It's an failure only if data is None
    if data is None:
        return FedMLResponse(code=ResponseCode.FAILURE, message=message)
    return FedMLResponse(code=ResponseCode.SUCCESS, message=message, data=data)


def get_metadata(data_name, api_key=None) -> FedMLResponse:
    api_key = authenticate(api_key)

    try:
        response = _get_dataset_metadata(api_key=api_key, data_name=data_name)
        code, message, data = _get_data_from_response(message="Failed to upload data", response=response)
    except Exception as e:
        return FedMLResponse(code=ResponseCode.FAILURE, message=f"Failed to get metadata of '{data_name}' with "
                                                                f"exception: {e}")

    if data:
        if data.get("datasetName", None) is None:
            message = f"Failed to get metadata of '{data_name}'. Data doesn't exists."
            logging.error(message)
            return FedMLResponse(code=ResponseCode.FAILURE, message=message)
        message = f"Successfully retrieved metadata for '{data_name}'."
        logging.info(message)
        return FedMLResponse(code=code, message=message, data=StorageMetadata(data))
    return FedMLResponse(code=code, message=message)


def list_objects(api_key=None) -> FedMLResponse:
    api_key = authenticate(api_key)
    try:
        response = _list_dataset(api_key=api_key)
    except Exception as e:
        message = f"Failed to list stored objects for account linked with api_key {api_key} with exception {e}"
        logging.error(message)
        return FedMLResponse(code=ResponseCode.FAILURE, message=message, data=None)

    code, message, data = _get_data_from_response(message=f"Failed to list storage objects", response=response)
    if data:
        storage_list = list()
        for data_obj in data:
            storage_list.append(StorageMetadata(data_obj))
        message = f"Successfully retrieved stored objects for account linked with api_key: {api_key}"
        logging.info(message)
        return FedMLResponse(code=ResponseCode.SUCCESS, message=message, data=storage_list)
    return FedMLResponse(code=code, message=message, data=data)


# Todo(alaydshah): Query service from object metadata. Make the transaction atomic or rollback if partially failed
def delete(data_name, service, api_key=None) -> FedMLResponse:
    api_key = authenticate(api_key)
    user_id, message = _get_user_id_from_api_key(api_key)

    if user_id is None:
        return FedMLResponse(code=ResponseCode.FAILURE, message=message)

    store = _get_storage_service(service)
    zip_file_name = data_name + ".zip"
    key = os.path.join(user_id, zip_file_name)
    result, message = store.delete_s3_zip(path_s3=key)
    if result:
        logging.info(f"Successfully deleted object from storage service.")
        try:
            response = _delete_dataset(api_key=api_key, data_name=data_name)
            code, message, data = _get_data_from_response(message="Failed to delete data", response=response)
        except Exception as e:
            message = (f"Deleted object from storage service but failed to delete object metadata from Nexus Backend "
                       f"with exception {e}")
            logging.error(message, data_name, service)
            return FedMLResponse(code=ResponseCode.FAILURE, message=message, data=False)
        if data:
            message = f"Successfully deleted '{data_name}' from storage service and object metadata from Nexus Backend"
            logging.info(message, data_name, service)
            return FedMLResponse(code=code, message=message, data=data)
        logging.error(message, data_name, service)
        return FedMLResponse(code=code, message=message, data=False)
    else:
        logging.error(message, data_name, service)
        return FedMLResponse(code=ResponseCode.FAILURE, message=message, data=False)


def _get_user_id_from_api_key(api_key: str) -> (str, str):
    user_url = ServerConstants.get_user_url()
    json_data = {
        "apiKey": api_key
    }
    response = ServerConstants.request(user_url, json_data)
    if response.status_code != 200:
        message = (f"Failed to get user info with response.status_code = {response.status_code}, "
                   f"response.content: {response.content}")
        logging.error(message)
        return None, message
    else:
        resp_data = response.json()
        code = resp_data.get("code", None)
        data = resp_data.get("data", None)
        if code is None or data is None or code == "FAILURE":
            message = resp_data.get("message", None)
            message = (f"Failed querying user info with following meesage: {message}, "
                       f"response.content: {response.content}")
            return None, message
    user_id = data.get("id", None)
    user_id = str(user_id) if user_id else None
    return user_id, f"Successfully fetched user id for api key {api_key}"


# Todo (alaydshah): Add time-expiring caching as this is a network call and config values don't change often
def _get_storage_service(service):
    configs = MLOpsConfigs.fetch_remote_storage_configs()
    if service == "R2":
        return S3Storage(configs[Configs.R2_CONFIG])
    elif service == "S3":
        return S3Storage(configs[Configs.S3_CONFIG])
    else:
        raise NotImplementedError(f"Service {service} not implemented")


def _archive_data(data_path: str) -> (str, str):
    src_local_path = os.path.abspath(data_path)
    root_dir = os.path.dirname(src_local_path)
    archive_name = os.path.basename(src_local_path) + '.zip'
    archive_path = os.path.join(root_dir, archive_name)
    try:
        # Create a zip archive
        shutil.make_archive(archive_path[:-4], 'zip', root_dir=root_dir, base_dir=os.path.basename(src_local_path))

        # Now you can use the archive_path for further processing or uploading
        return archive_path, f"Successfully archived data at: {archive_path}"
    except Exception as e:
        return None, f"Error archiving data: {e}"


def _create_dataset(api_key: str, json_data: dict) -> requests.Response:
    dataset_url = ServerConstants.get_dataset_url()
    cert_path = MLOpsConfigs.get_cert_path_with_version()
    headers = ServerConstants.API_HEADERS
    headers["Authorization"] = f"Bearer {api_key}"
    if cert_path is not None:
        try:
            requests.session().verify = cert_path
            response = requests.post(
                dataset_url, verify=True, headers=headers, json=json_data
            )
        except requests.exceptions.SSLError as err:
            MLOpsConfigs.install_root_ca_file()
            response = requests.post(
                dataset_url, verify=True, headers=headers, json=json_data
            )
    else:
        response = requests.post(
            dataset_url, verify=True, headers=headers, json=json_data
        )
    return response


def _list_dataset(api_key: str) -> requests.Response:
    list_dataset_url = ServerConstants.list_dataset_url()
    cert_path = MLOpsConfigs.get_cert_path_with_version()
    headers = ServerConstants.API_HEADERS
    headers["Authorization"] = f"Bearer {api_key}"
    if cert_path is None:
        try:
            requests.session().verify = cert_path
            response = requests.get(list_dataset_url, verify=True, headers=headers)
        except requests.exceptions.SSLError as err:
            MLOpsConfigs.install_root_ca_file()
            response = requests.get(list_dataset_url, verify=True, headers=headers)
    else:
        response = requests.get(list_dataset_url, verify=True, headers=headers)
    return response


def _get_dataset_metadata(api_key: str, data_name: str) -> requests.Response:
    dataset_metadata_url = ServerConstants.get_dataset_metadata_url()
    cert_path = MLOpsConfigs.get_cert_path_with_version()
    headers = ServerConstants.API_HEADERS
    headers["Authorization"] = f"Bearer {api_key}"
    if cert_path is not None:
        try:
            requests.session().verify = cert_path
            response = requests.get(
                dataset_metadata_url, verify=True, headers=headers,
                params={"datasetName": data_name}
            )
        except requests.exceptions.SSLError as err:
            MLOpsConfigs.install_root_ca_file()
            response = requests.get(
                dataset_metadata_url, verify=True, headers=headers,
                params={"datasetName": data_name}
            )
    else:
        response = requests.get(
            dataset_metadata_url, verify=True, headers=headers,
            params={"datasetName": data_name}
        )
    return response


def _delete_dataset(api_key: str, data_name: str) -> requests.Response:
    dataset_url = ServerConstants.get_dataset_url()
    cert_path = MLOpsConfigs.get_cert_path_with_version()
    headers = ServerConstants.API_HEADERS
    headers["Authorization"] = f"Bearer {api_key}"
    if cert_path is not None:
        try:
            requests.session().verify = cert_path
            response = requests.delete(
                dataset_url, verify=True, headers=headers,
                params={"datasetName": data_name}
            )
        except requests.exceptions.SSLError as err:
            MLOpsConfigs.install_root_ca_file()
            response = requests.delete(
                dataset_url, verify=True, headers=headers,
                params={"datasetName": data_name}
            )
    else:
        response = requests.delete(
            dataset_url, verify=True, headers=headers,
            params={"datasetName": data_name}
        )
    return response


def _get_data_from_response(message: str, response: requests.Response) -> (ResponseCode, str, object):
    if response.status_code != 200:
        message = f"{message} with response.status_code = {response.status_code}, response.content: {response.content}"
        return ResponseCode.FAILURE, message, None
    else:
        resp_data = response.json()
        code = resp_data.get("code", None)
        data = resp_data.get("data", None)
        message = resp_data.get("message", None)
        if code is None or data is None or code == "FAILURE":
            message = f"{message}, response.code = {code}, response.message: {message}"
            return ResponseCode.FAILURE, message, None

    return ResponseCode.SUCCESS, "Successfully parsed data from response", data
