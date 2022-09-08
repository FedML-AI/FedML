"""Utilities for file download and caching."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import logging
import logging.handlers
import multiprocessing.pool
import os
import shutil
import sys
import tempfile
import traceback
from typing import Dict
from typing import List
from typing import Set

from .utils import get_file

FED_GLD_SPLIT_FILE_BUNDLE = "landmarks-user-160k"
FED_GLD_SPLIT_FILE_DOWNLOAD_URL = (
    "http://storage.googleapis.com/gresearch/federated-vision-datasets/%s.zip"
    % FED_GLD_SPLIT_FILE_BUNDLE
)
FED_GLD_SPLIT_FILE_BUNDLE_MD5_CHECKSUM = "53c36bd7d5fc12f927af2820b7e4a57c"
FED_GLD_TRAIN_SPLIT_FILE = "federated_train.csv"
FED_GLD_TEST_SPLIT_FILE = "test.csv"
GLD_SHARD_BASE_URL = "https://s3.amazonaws.com/google-landmark"
NUM_SHARD_TRAIN = 500
MINI_GLD_TRAIN_DOWNLOAD_URL = (
    "https://storage.googleapis.com/tff-datasets-public/mini_gld_train_split.csv"
)
MINI_GLD_TRAIN_SPLIT_FILE = "mini_gld_train_split.csv"
MINI_GLD_TEST_DOWNLOAD_URL = (
    "https://storage.googleapis.com/tff-datasets-public/mini_gld_test.csv"
)
MINI_GLD_TEST_SPLIT_FILE = "mini_gld_test.csv"
MINI_GLD_TRAIN_SPLIT_FILE_MD5_CHECKSUM = "9fd62cf79a67046fdd673d3a0ac52841"
MINI_GLD_TEST_SPLIT_FILE_MD5_CHECKSUM = "298e9d19d66357236f66fe8e22920933"
FED_GLD_CACHE = "gld160k"
MINI_GLD_CACHE = "gld23k"
TRAIN_SUB_DIR = "train"
TEST_FILE_NAME = "test.tfRecord"
LOGGER = "gldv2"
KEY_IMAGE_BYTES = "image/encoded_jpeg"
KEY_IMAGE_DECODED = "image/decoded"
KEY_CLASS = "class"


def _listener_process(queue: multiprocessing.Queue, log_file: str):
    """Sets up a separate process for handling logging messages.
    This setup is required because without it, the logging messages will be
    duplicated when multiple processes are created for downloading GLD dataset.
    Args:
      queue: The queue to receive logging messages.
      log_file: The file which the messages will be written to.
    """
    root = logging.getLogger()
    h = logging.FileHandler(log_file)
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    h.setFormatter(fmt)
    root.addHandler(h)
    while True:
        try:
            record = queue.get()
            # We send None as signal to stop
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:  # pylint: disable=broad-except
            print("Something went wrong:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def _read_csv(path: str) -> List[Dict[str, str]]:
    """Reads a csv file, and returns the content inside a list of dictionaries.
    Args:
      path: The path to the csv file.
    Returns:
      A list of dictionaries. Each row in the csv file will be a list entry. The
      dictionary is keyed by the column names.
    """
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def _filter_images(shard: int, all_images: Set[str], image_dir: str, base_url: str):
    """Download full GLDv2 dataset, only keep images that are included in the federated gld v2 dataset.
    Args:
      shard: The shard of the GLDv2 dataset.
      all_images: A set which contains all images included in the federated GLD
        dataset.
      image_dir: The directory to keep all filtered images.
      base_url: The base url for downloading GLD v2 dataset images.
    Raises:
      IOError: when failed to download checksum.
    """
    shard_str = "%03d" % shard
    images_tar_url = "%s/train/images_%s.tar" % (base_url, shard_str)
    # images_md5_url = '%s/md5sum/train/md5.images_%s.txt' % (base_url, shard_str)
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = logging.getLogger(LOGGER)
        # logging.info('Start to download checksum for shard %s', shard_str)
        # get_file(
        #     'images_md5_%s.txt' % shard_str,
        #     origin=images_md5_url,
        #     cache_dir=tmp_dir)
        # logging.info('Downloaded checksum for shard %s successfully.', shard_str)
        logging.info("Start to download data for shard %s", shard_str)
        get_file(
            "images_%s.tar" % shard_str,
            origin=images_tar_url,
            extract=True,
            cache_dir=tmp_dir,
        )
        logging.info("Data for shard %s was downloaded successfully.", shard_str)
        count = 0
        for root, _, files in os.walk(tmp_dir):
            for filename in files:
                name, extension = os.path.splitext(filename)
                if extension == ".jpg" and name in all_images:
                    count += 1
                    shutil.copyfile(
                        os.path.join(root, filename), os.path.join(image_dir, filename)
                    )
        logging.info("Moved %d images from shard %s to %s", count, shard_str, image_dir)


def _download_data(num_worker: int, cache_dir: str, base_url: str):
    """
    Download the entire GLD v2 dataset, subset the dataset to only include the
    images in the federated GLD v2 dataset, and create both gld23k and gld160k
    datasets.
    Args:
      num_worker: The number of threads for downloading the GLD v2 dataset.
      cache_dir: The directory for caching temporary results.
      base_url: The base url for downloading GLD images.
    """
    logger = logging.getLogger(LOGGER)
    logging.info("Start to download fed gldv2 mapping files")

    path = get_file(
        "%s.zip" % FED_GLD_SPLIT_FILE_BUNDLE,
        origin=FED_GLD_SPLIT_FILE_DOWNLOAD_URL,
        extract=True,
        archive_format="zip",
        cache_dir=cache_dir,
    )

    get_file(
        MINI_GLD_TRAIN_SPLIT_FILE,
        origin=MINI_GLD_TRAIN_DOWNLOAD_URL,
        cache_dir=cache_dir,
    )
    get_file(
        MINI_GLD_TEST_SPLIT_FILE, origin=MINI_GLD_TEST_DOWNLOAD_URL, cache_dir=cache_dir
    )

    logging.info("Fed gldv2 mapping files are downloaded successfully.")
    base_path = os.path.dirname(path)
    train_path = os.path.join(
        base_path, FED_GLD_SPLIT_FILE_BUNDLE, FED_GLD_TRAIN_SPLIT_FILE
    )
    test_path = os.path.join(
        base_path, FED_GLD_SPLIT_FILE_BUNDLE, FED_GLD_TEST_SPLIT_FILE
    )
    train_mapping = _read_csv(train_path)
    test_mapping = _read_csv(test_path)
    all_images = set()
    all_images.update(
        [row["image_id"] for row in train_mapping],
        [row["image_id"] for row in test_mapping],
    )
    image_dir = os.path.join(cache_dir, "images")
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    logging.info("Start to download GLDv2 dataset.")
    with multiprocessing.pool.ThreadPool(num_worker) as pool:
        train_args = [
            (i, all_images, image_dir, base_url) for i in range(NUM_SHARD_TRAIN)
        ]
        pool.starmap(_filter_images, train_args)

    logging.info("Finish downloading GLDv2 dataset.")


def load_data(
    num_worker: int = 1,
    cache_dir: str = "cache",
    gld23k: bool = False,
    base_url: str = GLD_SHARD_BASE_URL,
):

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    q = multiprocessing.Queue(-1)
    listener = multiprocessing.Process(
        target=_listener_process, args=(q, os.path.join(cache_dir, "load_data.log"))
    )
    listener.start()
    logger = logging.getLogger(LOGGER)
    qh = logging.handlers.QueueHandler(q)
    logger.addHandler(qh)
    logging.info("Start to load data.")
    logging.info("Loading from cache failed, start to download the data.")

    _download_data(num_worker, cache_dir, base_url)

    # no return
    # fed_gld_train, fed_gld_test, mini_gld_train, mini_gld_test = _download_data(
    #       num_worker, cache_dir, base_url)

    q.put_nowait(None)
    listener.join()


if __name__ == "__main__":
    load_data(4, "cache")
