import os


def find_file_inside_folder(folder_path, file_name):
    """
    Recursively search for a file inside a folder and its sub-folders.
    return the full path of the file if found, otherwise return None.
    """
    for root, dirs, files in os.walk(folder_path):
        if file_name in files:
            return os.path.join(root, file_name)

    return None
