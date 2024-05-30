import logging
import os
import shutil

import requests
from fedml.api.modules.utils import authenticate
from fedml.core.distributed.communication.s3.remote_storage import S3Storage
from fedml.core.mlops.mlops_configs import Configs, MLOpsConfigs
from fedml.computing.scheduler.master.server_constants import ServerConstants
from fedml.api.fedml_response import FedMLResponse, ResponseCode

DUMMY_NEW_DATA = "new_data"
DUMMY_V1_EXISTS = "v1_exists"
DUMMY_MAIN_EXISTS = "main_exists"

class StorageMetadata(object):
    def __init__(self, data: dict):
        self.dataName = data.get("datasetName", None)
        self.description = data.get("description", None)
        self.tags = data.get("description", None)
        self.createdAt = data.get("createTime", None)
        self.updatedAt = data.get("updateTime", None)
        self.size = _get_size(data.get("fileSize",None))
        self.tag_list = data.get("tags", None)
        self.num_versions = 3


# Todo (alaydshah): Store service name in metadata
# Todo (alaydshah): If data already exists, don't upload again. Instead suggest to use update command

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
    
    response_version = _dummy_get_version(DUMMY_V1_EXISTS)
    data_already_exists = False
    new_version = False
    if response_version.data != None:
        data_already_exists = True
        choice = input("Do you want to overwrite the existing data? (y/n): ")
        if choice.lower() == 'n':
            print("A new version of the dataset will be created!\n")
            commit_message = input("Enter commit message or Description for version : \n")
            new_version = True
        elif choice.lower() == 'y':
            print("Data will be overwritten!\n")
        else:
            return FedMLResponse(code=ResponseCode.FAILURE, message="Invalid choice.")
    else:
        new_version = False
    version_to_write = ""
    if(data_already_exists):
        #means that the data exists already - overwriting or creating a new version. So the version list will exist.
        latest_version = response_version.data
        if(new_version):
            if(latest_version=="main"):
                version_to_write = "v_1"
            else:
                latest_version_number = int(latest_version.split("_")[1])
                new_version_number = latest_version_number + 1
                version_to_write = f"v_{new_version_number}"
        else:
            version_to_write = "main"
    else:
        #This is the first time data is being uploaded.
        version_to_write = "main"
        
    store = _get_storage_service(service)
    data_path = name
    name = os.path.splitext(os.path.basename(archive_path))[0] if name is None else name
    file_name = name + ".zip"
    dest_path = os.path.join(user_id,name,version_to_write,file_name)
    file_size = os.path.getsize(archive_path)

    file_uploaded_url = store.upload_file_with_progress(src_local_path=archive_path, dest_s3_path=dest_path,
                                                        show_progress=show_progress,
                                                        out_progress_to_err=out_progress_to_err,
                                                        progress_desc=progress_desc, metadata=metadata)
    os.remove(archive_path)
    if not file_uploaded_url:
        return FedMLResponse(code=ResponseCode.FAILURE, message=f"Failed to upload file: {archive_path}")
    
    description = ""
    tag_list = []
    if version_to_write == "main":
        tag_list = tag_list
        description = description
    else:
        tag_list = []
        description = commit_message
    
    if new_version:
        json_data ={
            "datasetName": name,
            "description":description,
            "fileSize": file_size,
            "fileUrl": file_uploaded_url,
            "tagNameList": tag_list,
            "version_id":version_to_write
        }
        try:
            response = _create_dataset_version(api_key=api_key, json_data=json_data)
            code, message, data = _dummy_get_data_from_response(response)
            #code, message, data = _get_data_from_response(message="Failed to upload data", response=response)
        except Exception as e:
            print("")
            #return FedMLResponse(code=ResponseCode.FAILURE, message=f"Failed to create dataset: {e}")

        if data:
            return FedMLResponse(code=code, message=message, data=file_uploaded_url)
        return FedMLResponse(code=code, message=message)

    else:
        # this part will update.
        json_data = {
            "datasetName": name,
            "description": description,
            "fileSize": file_size,
            "fileUrl": file_uploaded_url,
            "tagNameList": tag_list,
            "version_name":"main"
        }

        try:
            response = _update_dataset(api_key=api_key, json_data=json_data)
            #code, message, data = _get_data_from_response(message="Failed to update data", response=response)
            code, message, data = _dummy_get_data_from_response(response)
        except Exception as e:
            return FedMLResponse(code=ResponseCode.FAILURE, message=f"Failed to update dataset: {e}")

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

'''
Doesn't make the correct call as of now.
'''

def _update_dataset(api_key: str, json_data: dict) -> requests.Response:
    print("Update dataset function!")
    return FedMLResponse(code=ResponseCode.SUCCESS,message="Data updated successfully",data=json_data)

'''
Create version of the dataset
'''
def _create_dataset_version(api_key: str, json_data: dict) -> FedMLResponse:
    print("Create version of the dataset")
    print("The composite key in the backend can be datasetName + version_id")
    return FedMLResponse(code=ResponseCode.SUCCESS,message="Data version created successfully",data=json_data)


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

def _dummy_get_version(test_scenario:str=DUMMY_NEW_DATA):
    VERSION = None
    message = ""
    data = None
    if(test_scenario == DUMMY_NEW_DATA):
        message = "Data doesn't exist"
        data = None
    elif(test_scenario == DUMMY_MAIN_EXISTS):
        message = "Data exists"
        data = "main"
    elif(test_scenario == DUMMY_V1_EXISTS):
        message = "Data exists"
        data = "v_1"
    return FedMLResponse(code=ResponseCode.SUCCESS,message=message,data=data)

def _dummy_get_num_versions():
    NUM_VERSIONS = 1
    return NUM_VERSIONS

def _dummy_get_data_from_response(response:FedMLResponse):
    code, message, data = ResponseCode.SUCCESS, "Will get updated once re-routed to new create_dataset_with_version" , response.data
    return code, message, data 