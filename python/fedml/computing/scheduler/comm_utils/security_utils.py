import hashlib
import os

from fedml.computing.scheduler.scheduler_entry.constants import Constants

def get_file_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def get_content_hash(content):
    h = hashlib.md5(str(content).encode())  # hash content
    return h.hexdigest()  # return hash


def save_api_key(api_key):
    try:
        os.makedirs(Constants.get_secret_dir(), exist_ok=True)

        with open(Constants.get_launch_secret_file(), 'w') as secret_file_handle:
            secret_file_handle.writelines([api_key])
            secret_file_handle.close()
    except Exception as e:
        pass


def get_api_key():
    try:
        with open(Constants.get_launch_secret_file(), 'r') as secret_file_handle:
            api_key = secret_file_handle.readline()
            secret_file_handle.close()
            return api_key
    except Exception as e:
        return ""
