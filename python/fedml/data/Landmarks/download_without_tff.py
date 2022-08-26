import collections
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

import tensorflow as tf

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


def _create_dataset_with_mapping(
    image_dir: str, mapping: List[Dict[str, str]]
) -> List[tf.train.Example]:
    """Builds a dataset based on the mapping file and the images in the image dir.
    Args:
      image_dir: The directory contains the image files.
      mapping: A list of dictionaries. Each dictionary contains 'image_id' and
        'class' columns.
    Returns:
      A list of `tf.train.Example`.
    """
    logger = logging.getLogger(LOGGER)
    examples = []
    for row in mapping:
        img_path = os.path.join(image_dir, "%s.jpg" % row["image_id"])
        try:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
                examples.append(
                    tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                KEY_IMAGE_BYTES: tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[img_bytes])
                                ),
                                KEY_CLASS: tf.train.Feature(
                                    int64_list=tf.train.Int64List(
                                        value=[int(row["class"])]
                                    )
                                ),
                            }
                        )
                    )
                )
        except IOError as e:
            logging.warning("Image %s is not found. Exception: %s", img_path, e)
            continue
    return examples


def _create_train_data_files(cache_dir: str, image_dir: str, mapping_file: str):
    """Create the train data and persist it into a separate file per user.
    Args:
      cache_dir: The directory caching the intermediate results.
      image_dir: The directory containing all the downloaded images.
      mapping_file: The file containing 'image_id' to 'class' mappings.
    """
    logger = logging.getLogger(LOGGER)
    if not os.path.isdir(image_dir):
        logging.error("Image directory %s does not exist", image_dir)
        raise ValueError("%s does not exist or is not a directory" % image_dir)

    mapping_table = _read_csv(mapping_file)
    expected_cols = ["user_id", "image_id", "class"]
    if not all(col in mapping_table[0].keys() for col in expected_cols):
        logging.error("%s has wrong format.", mapping_file)
        raise ValueError(
            "The mapping file must contain user_id, image_id and class columns. "
            "The existing columns are %s" % ",".join(mapping_table[0].keys())
        )
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    mapping_per_user = collections.defaultdict(list)
    for row in mapping_table:
        user_id = row["user_id"]
        mapping_per_user[user_id].append(row)
    for user_id, data in mapping_per_user.items():
        examples = _create_dataset_with_mapping(image_dir, data)
        with tf.io.TFRecordWriter(os.path.join(cache_dir, str(user_id))) as writer:
            for example in examples:
                writer.write(example.SerializeToString())
            logging.info(
                "Created tfrecord file for user %s with %d examples.md, at %s",
                user_id,
                len(examples),
                cache_dir,
            )


def _create_test_data_file(cache_dir: str, image_dir: str, mapping_file: str):
    """Create the test data and persist it into a file.
    Args:
      cache_dir: The directory caching the intermediate results.
      image_dir: The directory containing all the downloaded images.
      mapping_file: The file containing 'image_id' to 'class' mappings.
    """
    logger = logging.getLogger(LOGGER)
    if not os.path.isdir(image_dir):
        logging.error("Image directory %s does not exist", image_dir)
        raise ValueError("%s does not exist or is not a directory" % image_dir)
    mapping_table = _read_csv(mapping_file)
    expected_cols = ["image_id", "class"]
    if not all(col in mapping_table[0].keys() for col in expected_cols):
        logging.error("%s has wrong format.", mapping_file)
        raise ValueError(
            "The mapping file must contain image_id and class columns. The existing"
            " columns are %s" % ",".join(mapping_table[0].keys())
        )
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    examples = _create_dataset_with_mapping(image_dir, mapping_table)
    with tf.io.TFRecordWriter(os.path.join(cache_dir, TEST_FILE_NAME)) as writer:
        for example in examples:
            writer.write(example.SerializeToString())
        logging.info("Created tfrecord file at %s", cache_dir)


def _create_federated_gld_dataset(
    cache_dir: str, image_dir: str, train_mapping_file: str, test_mapping_file: str
):
    """Generate fedreated GLDv2 dataset with the downloaded images.
    Args:
      cache_dir: The directory for caching the intermediate results.
      image_dir: The directory that contains the filtered images.
      train_mapping_file: The mapping file for the train set.
      test_mapping_file: The mapping file for the test set.
    """

    _create_train_data_files(
        cache_dir=os.path.join(cache_dir, FED_GLD_CACHE, TRAIN_SUB_DIR),
        image_dir=image_dir,
        mapping_file=train_mapping_file,
    )
    _create_test_data_file(
        cache_dir=os.path.join(cache_dir, FED_GLD_CACHE),
        image_dir=image_dir,
        mapping_file=test_mapping_file,
    )
    # return _load_data_from_cache(os.path.join(cache_dir, FED_GLD_CACHE))


def _create_mini_gld_dataset(cache_dir: str, image_dir: str):
    """Generate mini federated GLDv2 dataset with the downloaded images.
    Args:
      cache_dir: The directory for caching the intermediate results.
      image_dir: The directory that contains the filtered images.
    """
    train_path = tf.keras.utils.get_file(
        MINI_GLD_TRAIN_SPLIT_FILE,
        origin=MINI_GLD_TRAIN_DOWNLOAD_URL,
        file_hash=None,
        hash_algorithm="md5",
        cache_dir=cache_dir,
    )
    test_path = tf.keras.utils.get_file(
        MINI_GLD_TEST_SPLIT_FILE,
        origin=MINI_GLD_TEST_DOWNLOAD_URL,
        file_hash=None,
        hash_algorithm="md5",
        cache_dir=cache_dir,
    )
    _create_train_data_files(
        cache_dir=os.path.join(cache_dir, MINI_GLD_CACHE, TRAIN_SUB_DIR),
        image_dir=image_dir,
        mapping_file=train_path,
    )
    _create_test_data_file(
        cache_dir=os.path.join(cache_dir, MINI_GLD_CACHE),
        image_dir=image_dir,
        mapping_file=test_path,
    )


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
    images_md5_url = "%s/md5sum/train/md5.images_%s.txt" % (base_url, shard_str)
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger = logging.getLogger(LOGGER)
        logging.info("Start to download checksum for shard %s", shard_str)
        md5_path = tf.keras.utils.get_file(
            "images_md5_%s.txt" % shard_str, origin=images_md5_url, cache_dir=tmp_dir
        )
        with open(md5_path, "r") as f:
            md5_hash = f.read()
        if not md5_hash:
            msg = "Failed to download checksum for shard %s." % shard_str
            logging.info(msg)
            raise IOError(msg)
        logging.info("Downloaded checksum for shard %s successfully.", shard_str)
        logging.info("Start to download data for shard %s", shard_str)
        tf.keras.utils.get_file(
            "images_%s.tar" % shard_str,
            origin=images_tar_url,
            file_hash=None,
            hash_algorithm="md5",
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

    path = tf.keras.utils.get_file(
        "%s.zip" % FED_GLD_SPLIT_FILE_BUNDLE,
        origin=FED_GLD_SPLIT_FILE_DOWNLOAD_URL,
        file_hash=None,
        hash_algorithm="md5",
        extract=True,
        archive_format="zip",
        cache_dir=cache_dir,
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

    _create_federated_gld_dataset(cache_dir, image_dir, train_path, test_path)
    _create_mini_gld_dataset(cache_dir, image_dir)
    # no return
    # fed_gld_train, fed_gld_test = _create_federated_gld_dataset(
    #     cache_dir, image_dir, train_path, test_path)
    # mini_gld_train, mini_gld_test = _create_mini_gld_dataset(cache_dir, image_dir)

    # return fed_gld_train, fed_gld_test, mini_gld_train, mini_gld_test


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
