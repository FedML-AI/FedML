import logging
import os
import shutil

import requests
import math

import requests.exceptions
import tqdm
import sys
from fedml.api.modules.utils import authenticate
from fedml.core.distributed.communication.s3.remote_storage import S3Storage
from fedml.core.mlops.mlops_configs import Configs, MLOpsConfigs
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.api.fedml_response import FedMLResponse, ResponseCode


class StorageMetadata(object):
    def __init__(self, data: dict):
        self.dataName = data.get("datasetName", None)
        self.description = data.get("description", None)
        self.tags = data.get("description", None)
        self.createdAt = data.get("createTime", None)
        self.updatedAt = data.get("updateTime", None)
        self.size = _get_size(data.get("fileSize",None))
        self.tag_list = data.get("tags", None)
        self.download_url = data.get("fileUrl", None)


# Todo (alaydshah): Store service name in metadata
# Todo (alaydshah): If data already exists, don't upload again. Instead suggest to use update command
# Todo (bhargav) : Discuss and remove the service variable. Maybe needed sometime later.
def upload(data_path, api_key, name, description, tag_list, service, show_progress, out_progress_to_err, progress_desc,
           metadata) -> FedMLResponse:
    api_key = authenticate(api_key)

    user_id, message = _get_user_id_from_api_key(api_key)

    if user_id is None:
        return FedMLResponse(code=ResponseCode.FAILURE, message=message)
    
    if(not _check_data_path(data_path)):
        return FedMLResponse(code=ResponseCode.FAILURE,message="Invalid data path")

    archive_path, message = _archive_data(data_path)
    if not archive_path:
        return FedMLResponse(code=ResponseCode.FAILURE, message=message)

    name = os.path.splitext(os.path.basename(archive_path))[0] if name is None else name
    file_name = name + ".zip"
    dest_path = os.path.join(user_id, file_name)
    file_size = os.path.getsize(archive_path)

    file_uploaded_url, message = _upload_multipart(api_key, file_name, archive_path, show_progress,
                                                       out_progress_to_err,
                                                       progress_desc, metadata)


    os.remove(archive_path)
    if not file_uploaded_url:
        return FedMLResponse(code=ResponseCode.FAILURE, message=f"Failed to upload file: {archive_path}")

    json_data = {
        "datasetName": name,
        "description": description,
        "fileSize": file_size,
        "fileUrl": file_uploaded_url,
        "tagNameList": tag_list,
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

    metadata_response = get_metadata(data_name, api_key)
    if metadata_response.code == ResponseCode.SUCCESS:
        metadata = metadata_response.data
        if not metadata or not isinstance(metadata, StorageMetadata):
            error_message = f"Unable to get the download URL"
            logging.error(error_message)
            return FedMLResponse(code=ResponseCode.FAILURE, message=error_message)
        download_url = metadata.download_url
        zip_file_name = data_name + ".zip"
        path_local = os.path.abspath(zip_file_name)
        dest_path = os.path.abspath(dest_path) if dest_path else data_name
        if _download_using_presigned_url(download_url, zip_file_name, show_progress=show_progress):
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
            error_message = "Failed to download data from source"
            logging.error(error_message)
            return FedMLResponse(code=ResponseCode.FAILURE, message=error_message)

    else:
        error_message = metadata_response.message
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

def _get_num_chunks(file_size, max_chunk_size):
    num_chunks = math.ceil(file_size / max_chunk_size)
    return num_chunks


def get_chunks(file_path, chunk_size):
    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk


def _get_presigned_url(api_key, request_url, file_name, part_number=None):
    cert_path = MLOpsConfigs.get_cert_path_with_version()
    headers = ServerConstants.API_HEADERS
    headers["Authorization"] = f"Bearer {api_key}"
    params_dict = {'fileKey': file_name}
    if part_number is not None:
        params_dict['partNumber'] = part_number
    if cert_path is None:
        try:
            requests.session().verify = cert_path
            response = requests.get(request_url, verify=True, headers=headers, params=params_dict)
        except requests.exceptions.SSLError as err:
            MLOpsConfigs.install_root_ca_file()
            response = requests.get(request_url, verify=True, headers=headers, params=params_dict)
    else:
        response = requests.get(request_url, verify=True, headers=headers, params=params_dict)
    return response


def _upload_part(url,part_data,session):
    response = session.put(url,data=part_data,verify=True)
    return response


def _upload_chunk(presigned_url, chunk, part, pbar=None, max_retries=20,session=None):
    for retry_attempt in range(max_retries):
        try:
            response = _upload_part(presigned_url,chunk,session)
        except requests.exceptions.RequestException as e:
            if retry_attempt < max_retries:
                continue
            else:
                raise requests.exceptions.RequestException

        if(pbar is not None):
            pbar.update(chunk.__sizeof__())
        return {'etag': response.headers['ETag'], 'partNumber': part}
    raise requests.exceptions.RequestException

def _process_post_response(response):
    if response.status_code != 200:
        message = (f"Failed to complete multipart upload with status code = {response.status_code}, "
                   f"response.content: {response.content}")
        logging.error(message)
        return None, message
    else:
        resp_data = response.json()
        code = resp_data.get("code", None)
        data_url = resp_data.get("data", None)

        if code is None or data_url is None or code == "FAILURE":
            message = resp_data.get("message", None)
            message = (f"Failed to complete multipart upload with following message: {message}, "
                       f"response.content: {response.content}")
            return None, message

        return data_url, "Successfully uploaded the data! "


def _complete_multipart_upload(api_key, file_key, part_info, upload_id):
    complete_multipart_url = ServerConstants.get_complete_multipart_upload_url()
    body_dict = {"fileKey": file_key, 'partETags': part_info, 'uploadId': upload_id}

    cert_path = MLOpsConfigs.get_cert_path_with_version()
    headers = ServerConstants.API_HEADERS
    headers["Authorization"] = f"Bearer {api_key}"
    if cert_path is None:
        try:
            requests.session().verify = cert_path
            complete_multipart_response = requests.post(complete_multipart_url, json=body_dict, verify=True,
                                                        headers=headers)
        except requests.exceptions.SSLError as err:
            MLOpsConfigs.install_root_ca_file()
            complete_multipart_response = requests.post(complete_multipart_url, json=body_dict, verify=True,
                                                        headers=headers)
    else:
        complete_multipart_response = requests.post(complete_multipart_url, json=body_dict, verify=True,
                                                    headers=headers)

    return _process_post_response(complete_multipart_response)


def _upload_multipart(api_key: str, file_key, archive_path, show_progress, out_progress_to_err,
                      progress_desc_text, metadata):
    request_url = ServerConstants.get_presigned_multi_part_url()

    file_size = os.path.getsize(archive_path)

    max_chunk_size = 20 * 1024 * 1024

    num_chunks = _get_num_chunks(file_size, max_chunk_size)

    upload_id = ""
    presigned_urls = []

    presigned_url_response = _get_presigned_url(api_key, request_url, file_key, num_chunks)

    if presigned_url_response.status_code != 200:
        message = (f"Failed to get presigned URL with status code = {presigned_url_response.status_code}, "
                   f"response.content: {presigned_url_response.content}")
        logging.error(message)
        return None, message
    else:
        resp_data = presigned_url_response.json()
        code = resp_data.get("code", None)
        data = resp_data.get("data", None)

        if code is None or data is None or code == "FAILURE":
            message = resp_data.get("message", None)
            message = (f"Failed getting presigned URL with following message: {message}, "
                       f"response.content: {presigned_url_response.content}")
            return None, message

        upload_id = data['uploadId']
        presigned_urls = data['urls']

    parts = []
    chunks = get_chunks(archive_path, max_chunk_size)
    part_info = []
    chunk_count = 0
    successful_chunks = 0
    #TODO: (bhargav191098) Using Thread pool and confirming openssl issue
    atomic_session = requests.session()
    atomic_session.verify = MLOpsConfigs.get_cert_path_with_version()
    with tqdm.tqdm(total=file_size, unit="B", unit_scale=True,
                   file=sys.stderr if out_progress_to_err else sys.stdout,
                   desc=progress_desc_text, leave=False) as pbar:
        for part, chunk in enumerate(chunks, start=1):
            presigned_url = presigned_urls[part - 1]
            chunk_count += 1
            if show_progress:
                try:
                    part_data = _upload_chunk(presigned_url=presigned_url, chunk=chunk, part=part,
                                             pbar=pbar,session=atomic_session)
                    part_info.append(part_data)
                    successful_chunks += 1
                except Exception as e:
                    return None, "unsuccessful"

            else:
                try:
                    part_data = _upload_chunk(presigned_url=presigned_url, chunk=chunk, part=part,
                                             pbar=pbar,session=atomic_session)
                    part_info.append(part_data)
                    successful_chunks += 1
                except Exception as e:
                    return None, "unsuccessful"

    if successful_chunks == chunk_count:
        return _complete_multipart_upload(api_key, file_key, part_info, upload_id)
    else:
        return None, "Unsuccessful!"


def _download_using_presigned_url(url, fname, chunk_size=1024 * 1024, show_progress=True):
    download_response = requests.get(url, verify=True, stream=True)
    if download_response.status_code == 200:
        total = int(download_response.headers.get('content-length', 0))
        if show_progress:
            with open(fname, 'wb') as file, tqdm.tqdm(
                    desc=fname,
                    total=total,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as bar:
                for data in download_response.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)
        else:
            with open(fname, "wb") as file:
                for data in download_response.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
        return True
    return False

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

def _check_data_path(data_path):
    if os.path.isdir(data_path) or os.path.isfile(data_path):
        return True
    return False


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

def _get_size(size_in_bytes:str)->str:
    size_str = ""
    if(size_in_bytes):
        size = int(size_in_bytes)
        size_in_gb = size / (1024 * 1024 * 1024)
        size_in_mb = size / (1024 * 1024)
        size_in_kb = size / 1024
        if(size_in_gb >= 1):
            size_str = f"{size_in_gb:.2f} GB"
        elif(size_in_mb >= 1):
            size_str = f"{size_in_mb:.2f} MB"
        elif(size_in_kb >= 1):
            size_str = f"{size_in_kb:.2f} KB"
        else:
            size_str = f"{size} B"
    return size_str 