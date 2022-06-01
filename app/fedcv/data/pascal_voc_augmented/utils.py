import sys
import os
import requests
import tarfile
import logging


def __convert_size(size_in_bytes, unit):
    """
  Converts the bytes to human readable size format.

  Args:
    size_in_bytes (int): The number of bytes to convert
    unit (str): The unit to convert to.
  """
    if unit == 'GB':
        return '{:.2f} GB'.format(size_in_bytes / (1024 * 1024 * 1024))
    elif unit == 'MB':
        return '{:.2f} MB'.format(size_in_bytes / (1024 * 1024))
    elif unit == 'KB':
        return '{:.2f} KB'.format(size_in_bytes / 1024)
    else:
        return '{:.2f} bytes'.format(size_in_bytes)


def _download_file(name, url, file_path, unit):
    """
  Downloads the file to the path specified

  Args:
    name (str): The name to print in console while downloading.
    url (str): The url to download the file from.
    file_path (str): The local path where the file should be saved.
    unit (str): The unit to convert to.
  """
    with open(file_path, 'wb') as f:
        logging.info('Downloading {}...'.format(name))
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise EnvironmentError('Encountered error while fetching. Status Code: {}, Error: {}'.format(response.status_code,
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


def _extract_file(file_path, extract_dir):
    """
  Extracts the file to the specified path.

  Args:
    file_path (str): The local path where the zip file is located.
    extract_dir (str): The local path where the files must be extracted.
  """
    with tarfile.open(file_path) as tar:
        logging.info('Extracting {} to {}...'.format(file_path, extract_dir))
        tar.extractall(path=extract_dir)
        tar.close()
        os.remove(file_path)
        logging.info('Extracted {}'.format(file_path))
