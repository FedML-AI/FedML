import concurrent.futures
import json
import logging
import os
import shutil
import sys

import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
import math
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
        self.size = _get_size(data.get("fileSize", None))
        self.tag_list = data.get("tags", None)


# Todo (alaydshah): Store service name in metadata
# Todo (alaydshah): If data already exists, don't upload again. Instead suggest to use update command

def upload(data_path, api_key, name, description, tag_list, service, show_progress, out_progress_to_err, progress_desc,
           metadata) -> FedMLResponse:
    api_key = authenticate(api_key)

    user_id, message = _get_user_id_from_api_key(api_key)

    if user_id is None:
        return FedMLResponse(code=ResponseCode.FAILURE, message=message)

    if (not _check_data_path(data_path)):
        return FedMLResponse(code=ResponseCode.FAILURE, message="Invalid data path")

    archive_path, message = _archive_data(data_path)
    if not archive_path:
        return FedMLResponse(code=ResponseCode.FAILURE, message=message)

    store = _get_storage_service(service)
    name = os.path.splitext(os.path.basename(archive_path))[0] if name is None else name
    file_name = name + ".zip"
    #dest_path = os.path.join(user_id, file_name)
    file_size = os.path.getsize(archive_path)

    size_in_mb = file_size / (1024 * 1024)
    #print("Size in MB ",size_in_mb)

    if size_in_mb < 10:
        print("As a single upload! ")
        file_uploaded_url, message = _upload_as_single_file(api_key, file_name, archive_path, show_progress, out_progress_to_err,
                                                            progress_desc, metadata)
    else:
        #num_chunks = _get_num_chunks(file_size)
        file_uploaded_url, message = _upload_multipart(api_key, file_name, archive_path, show_progress, out_progress_to_err,
                                                   progress_desc, metadata)

    # file_uploaded_url = store.upload_file_with_progress(src_local_path=archive_path, dest_s3_path=dest_path,
    #                                                     show_progress=show_progress,
    #                                                     out_progress_to_err=out_progress_to_err,
    #                                                     progress_desc=progress_desc, metadata=metadata)
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


def _upload_as_single_file(api_key: str, file_key: str, archive_path, show_progress, out_progress_to_err,
                           progress_desc_text, metadata):
    single_presigned_request_url = ServerConstants.get_presigned_single_part_url()
    presigned_url_response = _get_presigned_url(api_key=api_key,request_url=single_presigned_request_url, file_name=file_key)

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
            message = (f"Failed getting presigned URL with following meesage: {message}, "
                       f"response.content: {presigned_url_response.content}")
            return None, message

    upload_url = data.get("uploadUrl")
    download_url = data.get("downloadUrl")

    with open(archive_path, "rb") as f:
        old_file_position = f.tell()
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(old_file_position, os.SEEK_SET)
        data = f.read()
        if not data:
            return
        if show_progress:
            with tqdm.tqdm(total=file_size, unit="B", unit_scale=True,
                           file=sys.stderr if out_progress_to_err else sys.stdout,
                           desc=progress_desc_text) as pbar:
                response = _upload_file(upload_url, f)

                pbar.update(file_size)
                return _process_put_response_placeholder(response, download_url)
        else:
            response = _upload_file(upload_url, f)
            return _process_put_response_placeholder(response, download_url)


def get_chunks(file_path, chunk_size):
    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk

def upload_chunk(presigned_url, chunk,part, pbar, max_retries = 3):
    for retry_attempt in range(max_retries):
        cert_path = MLOpsConfigs.get_cert_path_with_version()
        if cert_path is None:
            try:
                requests.session().verify = cert_path
                response = _upload_part(presigned_url, chunk)
            except requests.exceptions.RequestException as e:
                if retry_attempt < max_retries:
                    continue
                else:
                    #print("Ran out of retries ")
                    raise requests.exceptions.RequestException
            response.raise_for_status()
            return {'etag': response.headers['ETag'], 'partNumber': part}
        else:
            try:
                response = _upload_part(presigned_url,chunk)
            except Exception as e:
                if retry_attempt < max_retries:
                    continue
                else:
                    # Ran out of retries
                    raise requests.exceptions.RequestException
            response.raise_for_status()
            pbar.update(chunk.__sizeof__())
            return {'etag': response.headers['ETag'], 'partNumber': part}

    raise requests.exceptions.RequestException



def _upload_multipart(api_key: str, file_key, archive_path, show_progress, out_progress_to_err,
                           progress_desc_text, metadata):
    request_url = ServerConstants.get_presigned_multi_part_url()

    file_size = os.path.getsize(archive_path)

    max_chunk_size = 8 * 1024 * 1024

    num_chunks = _get_num_chunks(file_size,max_chunk_size)

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
        chunks = get_chunks(archive_path,max_chunk_size)
        part_info = []
        chunk_count = 0
        successful_chunks = 0

        with tqdm.tqdm(total=file_size, unit="B", unit_scale=True,
                           file=sys.stderr if out_progress_to_err else sys.stdout,
                           desc=progress_desc_text) as pbar:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for part,chunk in enumerate(chunks,start=1):
                    presigned_url = presigned_urls[part-1]
                    chunk_count += 1
                    # Upload chunk to presigned_url in a separate thread from the thread pool of 10 workers.
                    future = executor.submit(upload_chunk,presigned_url=presigned_url,chunk=chunk,part=part,pbar=pbar)
                    futures.append(future)

                # Wait for all uploads to complete
                done,not_done = concurrent.futures.wait(futures)
                for future in done:
                    try:
                        part_info.append(future.result())
                        successful_chunks += 1
                    except Exception as e:
                        break

        #print("Number of parts succeeded : ",successful_chunks)
        if successful_chunks == chunk_count:
            return _complete_multipart_upload(api_key,file_key,part_info,upload_id)
        else:
            return None, "Unsuccessful!"


def _complete_multipart_upload(api_key,file_key,part_info,upload_id):
    complete_multipart_url = ServerConstants.get_complete_multipart_upload_url()
    body_dict = {"fileKey": file_key, 'partETags': part_info, 'uploadId': upload_id}

    cert_path = MLOpsConfigs.get_cert_path_with_version()
    headers = ServerConstants.API_HEADERS
    headers["Authorization"] = f"Bearer {api_key}"
    if cert_path is None:
        try:
            requests.session().verify = cert_path
            complete_multipart_response = requests.post(complete_multipart_url, json=body_dict,verify=True,headers=headers)
        except requests.exceptions.SSLError as err:
            MLOpsConfigs.install_root_ca_file()
            complete_multipart_response = requests.post(complete_multipart_url, json=body_dict, verify=True,headers=headers)
    else:
        complete_multipart_response = requests.post(complete_multipart_url, json=body_dict, verify=True, headers =headers)

    return _process_post_response(complete_multipart_response)


def _process_put_response_placeholder(response, download_url=None):
    if response.status_code != 200:
        message = (f"Failed to upload data with status code = {response.status_code}, "
                   f"response.content: {response.content}")
        logging.error(message)
        return None, message
    else:
        code = response.status_code
        if code is None or code == "FAILURE":
            message = (f"Failed to upload data, "
                       f"response.content: {response.content}")
            return None, message
        return download_url, "Successfully uploaded the data! "


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


def _upload_part(url, part_data):
    response = requests.put(url, data=part_data)
    return response

def _upload_file(presigned_url, file_data):
    response = requests.put(presigned_url, files=file_data)
    return response


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


def _get_size(size_in_bytes: str) -> str:
    size_str = ""
    if (size_in_bytes):
        size = int(size_in_bytes)
        size_in_gb = size / (1024 * 1024 * 1024)
        size_in_mb = size / (1024 * 1024)
        size_in_kb = size / 1024
        if (size_in_gb >= 1):
            size_str = f"{size_in_gb:.2f} GB"
        elif (size_in_mb >= 1):
            size_str = f"{size_in_mb:.2f} MB"
        elif (size_in_kb >= 1):
            size_str = f"{size_in_kb:.2f} KB"
        else:
            size_str = f"{size} B"
    return size_str


def _get_num_chunks(file_size, max_chunk_size):
    num_chunks = math.ceil(file_size / max_chunk_size)
    return num_chunks
