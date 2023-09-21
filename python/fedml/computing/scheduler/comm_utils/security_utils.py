import hashlib
import os


def get_file_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update("".join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def get_content_hash(content):
    h = hashlib.md5(str(content).encode())  # hash content
    return h.hexdigest()  # return hash
