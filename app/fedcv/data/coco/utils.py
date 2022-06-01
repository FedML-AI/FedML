import sys
import requests
import os
import logging

from pathlib import PurePath
from zipfile import ZipFile


def __convert_size(size_in_bytes: int, unit: str) -> str:
    """
    Converts the bytes to human readable size format.

    Args:
        size_in_bytes: The number of bytes to convert
        unit: The unit to convert to.

    Returns:
        The converted size string.
    """
    if unit == 'GB':
        return '{:.2f} GB'.format(size_in_bytes / (1024 * 1024 * 1024))
    elif unit == 'MB':
        return '{:.2f} MB'.format(size_in_bytes / (1024 * 1024))
    elif unit == 'KB':
        return '{:.2f} KB'.format(size_in_bytes / 1024)
    else:
        return '{:.2f} bytes'.format(size_in_bytes)


def _download_file(name: str, url: str, file_path: PurePath, unit: str) -> None:
    """
    Downloads the file to the path specified

    Args:
        name: The name to print in console while downloading.
        url: The url to download the file from.
        file_path: The local path where the file should be saved.
        unit: The unit to convert to.
    """
    with open(file_path, 'wb') as f:
        logging.info('Downloading {}...'.format(name))
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise EnvironmentError(
                'Encountered error while fetching. Status Code: {}, Error: {}'.format(response.status_code,
                                                                                      response.content))
        total = response.headers.get('content-length')
        human_readable_total = __convert_size(int(total), unit)

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                human_readable_downloaded = __convert_size(int(downloaded), unit)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write(
                    '\r[{}{}] {}% ({}/{})'.format('#' * done, '.' * (50 - done), int((downloaded / total) * 100),
                                                  human_readable_downloaded, human_readable_total))
                sys.stdout.flush()
    sys.stdout.write('\n')
    logging.info('Download Completed.')


def _extract_file(file_path: PurePath, extract_dir: PurePath) -> None:
    """
    Extracts the file to the specified path.

    Args:
        file_path: The local path where the zip file is located.
        extract_dir: The local path where the files must be extracted.
    """
    with ZipFile(file_path, 'r') as zip_file:
        logging.info('Extracting {} to {}...'.format(file_path, extract_dir))
        zip_file.extractall(extract_dir)
        zip_file.close()
        os.remove(file_path)
        logging.info('Extracted {}'.format(file_path))
